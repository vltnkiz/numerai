#!/usr/bin/env python3
"""LIVE entrypoint: run the pipeline, check the gate, AND SUBMIT.

Every invocation uploads to SUBMISSION_MODEL_SLOT, overwriting the round's real
submission. To measure without submitting, use scripts/run_experiment.py.

Usage:
  python scripts/run_pipeline.py --model linear
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone

from zemir.config import (
    LIVE,
    MIN_VALIDATION_MEAN_CORR,
    MODEL_NAMES,
    MODEL_WEIGHTS,
    NEUTRALIZERS,
    SUBMISSION_MODEL_SLOT,
    build_trainers,
)
from zemir.pipeline import (
    ValidationScoreBelowThreshold,
    append_score_log,
    run_pipeline,
    submit_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="linear")
    args = parser.parse_args()

    config = replace(
        LIVE, model_weights=MODEL_WEIGHTS[args.model], neutralizers=NEUTRALIZERS[args.model]
    )
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
    submission = None
    if not below_gate:
        submission = submit_predictions(
            result.live_predictions_neutralized.rename("prediction"),
            model_slot=SUBMISSION_MODEL_SLOT,
            run_dir=result.run_dir,
        )
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
        raise ValidationScoreBelowThreshold(
            f"validation mean_corr {result.combined_validation_score.mean_corr:.4f} < "
            f"MIN_VALIDATION_MEAN_CORR {MIN_VALIDATION_MEAN_CORR:.4f} — not submitting"
        )


if __name__ == "__main__":
    main()
