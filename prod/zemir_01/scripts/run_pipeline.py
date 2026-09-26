#!/usr/bin/env python3
"""LIVE entrypoint: run the pipeline, check the gate, AND SUBMIT.

Every invocation that finds an unfinished round uploads to SUBMISSION_MODEL_SLOT,
overwriting the round's real submission. To measure without submitting, use
scripts/run_experiment.py.

Started by prod/zemir_01/scheduling/Invoke-ZemirLiveRun.ps1 (issue #69), which
maps the exit codes below to failure issues. An invocation with nothing to do
exits 0.

Takes no strategy argument: `PRODUCTION_STRATEGY` is the single line that
decides what production ships. A flag here would be a second place that
decision could be made, and the two could silently disagree — see
docs/adr/0001-the-live-entrypoint-runs-one-named-strategy.md.

Usage:
  python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone

from numerapi import NumerAPI

from zemir.config import (
    LIVE,
    MIN_AVAILABLE_MEMORY_GIB,
    MIN_VALIDATION_MEAN_CORR,
    PRODUCTION_STRATEGY,
    SUBMISSION_MODEL_SLOT,
)
from zemir.data import download, load_meta_model, refresh_validation, resolve_feature_sets
from zemir.pipeline import (
    append_score_log,
    feature_set_names,
    format_gate,
    format_run_scores,
    record_gate,
    run_columns,
    run_pipeline,
    score_run,
    submit_predictions,
)
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
# The run got as far as it was going to (submitted, or stopped by the gate or
# a round change), but scoring it afterwards failed. Never means the upload
# failed: an upload failure raises before this is reachable (issue #97).
EXIT_POST_SUBMISSION_SCORING_FAILED = 6


def main() -> int:
    config = LIVE
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

    try:
        require_available_memory(MIN_AVAILABLE_MEMORY_GIB)
    except InsufficientMemory as exc:
        print(f"not starting: {exc}")
        return EXIT_INSUFFICIENT_MEMORY

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_sets = resolve_feature_sets(
        config.data_version, feature_set_names(config, PRODUCTION_STRATEGY)
    )
    dataset = download(
        config.data_version,
        run_columns(config, PRODUCTION_STRATEGY, feature_sets),
        target_column=config.target_column,
    )
    result = run_pipeline(
        config,
        run_id=run_id,
        strategy=PRODUCTION_STRATEGY,
        feature_sets=feature_sets,
        dataset=dataset,
    )

    print(f"run_id: {result.run_id}")
    print(f"live predictions: {len(result.live_predictions)} rows")

    # The gate guards submission, so it sits with submission — not inside the
    # pipeline, which cannot submit and so has nothing to stop. Its number is
    # the only score computed before the upload (issue #97).
    gate = record_gate(result, threshold=MIN_VALIDATION_MEAN_CORR)
    below_gate = not gate["passed"]
    print(format_gate(result, threshold=MIN_VALIDATION_MEAN_CORR))
    # A new round opening mid-fit would receive predictions made from the
    # previous round's live features. Not a final outcome: the wrapper reruns
    # for the new round. No current round at all (a between-rounds gap or an
    # API error) is not a change — the upload then fails loudly instead of the
    # run skipping quietly.
    round_changed = False
    submission = None
    if not below_gate:
        current = fetch_current_round(napi)
        round_changed = current is not None and current.number != live_round.number
        if not round_changed:
            # Checked here, not before the fit: a fit can run past staking close.
            if live_round.is_late(datetime.now(timezone.utc)):
                print(
                    f"WARNING: round {live_round.number} staking closed at "
                    f"{live_round.close_staking_time.isoformat()} — submitting late "
                    "(scored, unstaked)"
                )
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

    # Everything below is recorded, never decided on, so nothing in it may
    # take the submission down with it (issue #97). The meta model is
    # refreshed here, after the upload, so the harness reads a copy no older
    # than today's (issue #101). A failed refresh already falls back to the
    # cached copy; only the no-cache case raises.
    scores = None
    try:
        meta_model = load_meta_model(config.data_version, refresh=True)
        scores = score_run(
            result,
            meta_model=meta_model,
            scoring_universe=feature_sets[config.feature_set],
        )
        print(format_run_scores(scores))
    except Exception:
        traceback.print_exc()
        print(
            "ERROR: post-submission scoring failed. "
            + (
                f"The submission itself went through (submission_id={submission.submission_id})."
                if submission
                else "Nothing was submitted, for the reason given below, not because of this."
            )
        )

    # Logged regardless of the gate outcome (issue #68): the gate only catches
    # a run crashing through the floor, not one quietly declining above it —
    # that needs history, which a gate-only run would never accumulate.
    append_score_log(
        run_id=result.run_id,
        target_column=config.target_column,
        gate=gate,
        scores=scores,
        submission_id=submission.submission_id if submission else None,
    )
    exit_code = 0
    if below_gate:
        # Deterministic fit on unchanged data: a retry would fail identically.
        record_round_outcome(live_round.number, GATE_FAILED, run_id=result.run_id)
        print("not submitting: below the gate")
        exit_code = EXIT_GATE_FAILED
    elif round_changed:
        print(
            f"round {live_round.number} is no longer current — not submitting its "
            "predictions into the next round"
        )
        exit_code = EXIT_ROUND_CHANGED
    elif scores is None:
        exit_code = EXIT_POST_SUBMISSION_SCORING_FAILED

    # Last, after every outcome is recorded, and never on a round change: the
    # wrapper reruns straight away for the new round, and a 5.6 GB download
    # would sit in front of that upload. Keeps the next run's gate frame
    # current; a failure only warns.
    if not round_changed:
        refresh_validation(config.data_version)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
