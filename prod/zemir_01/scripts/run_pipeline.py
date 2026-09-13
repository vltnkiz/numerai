#!/usr/bin/env python3
"""LIVE entrypoint: run the pipeline, check the gate, AND SUBMIT.

Every invocation that finds an unfinished round uploads to SUBMISSION_MODEL_SLOT,
overwriting the round's real submission. To measure without submitting, use
scripts/run_experiment.py.

Started by prod/zemir_01/scheduling/Invoke-ZemirLiveRun.ps1 (issue #69), which
maps the exit codes below to failure issues. An invocation with nothing to do
exits 0.

Usage:
  python scripts/run_pipeline.py --model ensemble
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone

from numerapi import NumerAPI

from zemir.config import (
    LIVE,
    MIN_AVAILABLE_MEMORY_GIB,
    MIN_VALIDATION_MEAN_CORR,
    MODEL_NAMES,
    MODEL_WEIGHTS,
    NEUTRALIZERS,
    SUBMISSION_MODEL_SLOT,
    build_trainers,
)
from zemir.pipeline import append_score_log, run_pipeline, submit_predictions
from zemir.schedule import (
    GATE_FAILED,
    SUBMITTED,
    InsufficientMemory,
    RoundNotOpen,
    fetch_current_round,
    record_round_outcome,
    require_available_memory,
    round_to_run,
)

# Read by Invoke-ZemirLiveRun.ps1. Anything else non-zero is a crash.
EXIT_GATE_FAILED = 2
EXIT_ROUND_NOT_OPEN = 3
EXIT_INSUFFICIENT_MEMORY = 4
EXIT_ROUND_CHANGED = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="linear")
    args = parser.parse_args()

    config = replace(
        LIVE, model_weights=MODEL_WEIGHTS[args.model], neutralizers=NEUTRALIZERS[args.model]
    )
    napi = NumerAPI()

    # Issue #33/#69: decide from Numerai's round state before downloading live
    # data, rather than trusting whichever trigger started this to be on time.
    try:
        live_round = round_to_run(napi=napi)
    except RoundNotOpen as exc:
        print(f"round not open: {exc}")
        return EXIT_ROUND_NOT_OPEN
    if live_round is None:
        print("nothing to do: no opened round left unfinished, and outside the wait window")
        return 0
    # The wrapper keys failure issues on this line.
    print(f"ZEMIR_ROUND={live_round.number}")
    if live_round.is_late(datetime.now(timezone.utc)):
        print(
            f"WARNING: round {live_round.number} staking closed at "
            f"{live_round.close_staking_time.isoformat()} — submitting late (scored, unstaked)"
        )

    try:
        require_available_memory(MIN_AVAILABLE_MEMORY_GIB)
    except InsufficientMemory as exc:
        print(f"not starting: {exc}")
        return EXIT_INSUFFICIENT_MEMORY

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = run_pipeline(
        config,
        run_id=run_id,
        trainers=build_trainers(args.model, config),
    )

    print(f"run_id: {result.run_id}")
    for name, score in result.validation_scores.items():
        print(f"  {name}: mean_corr={score.mean_corr:.4f}  sharpe={score.sharpe:.4f}")
    print(
        f"combined validation mean_corr: {result.combined_validation_score.mean_corr:.4f}  "
        f"sharpe: {result.combined_validation_score.sharpe:.4f}"
    )
    print(f"live predictions: {len(result.live_predictions)} rows")

    # The gate guards submission, so it sits with submission — not inside the
    # pipeline, which cannot submit and so has nothing to stop.
    below_gate = result.combined_validation_score.mean_corr < MIN_VALIDATION_MEAN_CORR
    # A new round opening mid-fit would receive predictions made from the
    # previous round's live features. Not a final outcome: the next trigger
    # runs the new round from scratch.
    round_changed = False
    submission = None
    if not below_gate:
        current = fetch_current_round(napi)
        round_changed = current is None or current.number != live_round.number
        if not round_changed:
            submission = submit_predictions(
                result.live_predictions_neutralized.rename("prediction"),
                model_slot=SUBMISSION_MODEL_SLOT,
                run_dir=result.run_dir,
            )
            record_round_outcome(live_round.number, SUBMITTED, run_id=result.run_id)
            print(
                f"submitted to {submission.model_name} ({submission.model_id}): "
                f"submission_id={submission.submission_id}"
            )

    # Logged regardless of the gate outcome (issue #68): the gate only catches
    # a run crashing through the floor, not one quietly declining above it —
    # that needs history, which a gate-only run would never accumulate.
    append_score_log(
        run_id=result.run_id,
        target_column=config.target_column,
        validation_scores=result.validation_scores,
        combined_validation_score=result.combined_validation_score,
        submission_id=submission.submission_id if submission else None,
    )
    if below_gate:
        # Deterministic fit on unchanged data: a retry would fail identically.
        record_round_outcome(live_round.number, GATE_FAILED, run_id=result.run_id)
        print(
            f"validation mean_corr {result.combined_validation_score.mean_corr:.4f} < "
            f"MIN_VALIDATION_MEAN_CORR {MIN_VALIDATION_MEAN_CORR:.4f} — not submitting"
        )
        return EXIT_GATE_FAILED
    if round_changed:
        print(
            f"round {live_round.number} is no longer current — not submitting its "
            "predictions into the next round"
        )
        return EXIT_ROUND_CHANGED
    return 0


if __name__ == "__main__":
    sys.exit(main())
