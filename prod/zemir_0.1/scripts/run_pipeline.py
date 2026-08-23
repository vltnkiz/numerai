#!/usr/bin/env python3
"""CLI entrypoint: run one end-to-end zemir pipeline invocation.

Usage:
  python scripts/run_pipeline.py --model linear
  python scripts/run_pipeline.py --model era_boost
  python scripts/run_pipeline.py --model ensemble
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from zemir.models import train_linear, train_xgboost
from zemir.pipeline import Trainer, run_pipeline

DATA_VERSION = "5.0"
FEATURE_SET = "small"
NEUTRALIZATION_PROPORTION = 0.5
MIN_VALIDATION_MEAN_CORR = 0.0
SUBMISSION_MODEL_SLOT = "zemir_01"

XGBOOST_HYPERPARAMS = dict(
    trees_per_step=50,
    num_iters=40,
    proportion=0.5,
    learning_rate=0.01,
    max_depth=5,
    colsample_bytree=0.1,
)

TRAINERS: dict[str, dict[str, Trainer]] = {
    "linear": {"linear": lambda X, y, era: train_linear(X, y)},
    "era_boost": {
        "era_boost": lambda X, y, era: train_xgboost(X, y, era, **XGBOOST_HYPERPARAMS)
    },
    "ensemble": {
        "linear": lambda X, y, era: train_linear(X, y),
        "era_boost": lambda X, y, era: train_xgboost(X, y, era, **XGBOOST_HYPERPARAMS),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(TRAINERS), default="linear")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = run_pipeline(
        run_id=run_id,
        data_version=DATA_VERSION,
        feature_set=FEATURE_SET,
        neutralization_proportion=NEUTRALIZATION_PROPORTION,
        min_validation_mean_corr=MIN_VALIDATION_MEAN_CORR,
        submission_model_slot=SUBMISSION_MODEL_SLOT,
        trainers=TRAINERS[args.model],
    )

    print(f"run_id: {result.run_id}")
    for name, score in result.validation_scores.items():
        print(f"  {name}: mean_corr={score.mean_corr:.4f}  sharpe={score.sharpe:.4f}")
    print(
        f"combined validation mean_corr: {result.combined_validation_score.mean_corr:.4f}  "
        f"sharpe: {result.combined_validation_score.sharpe:.4f}"
    )
    print(f"live predictions: {len(result.live_predictions)} rows")
    print(
        f"submitted to {result.submission.model_name} ({result.submission.model_id}): "
        f"submission_id={result.submission.submission_id}"
    )


if __name__ == "__main__":
    main()
