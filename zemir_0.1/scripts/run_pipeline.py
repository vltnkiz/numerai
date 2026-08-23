#!/usr/bin/env python3
"""CLI entrypoint: run one end-to-end zemir pipeline invocation.

Usage: python scripts/run_pipeline.py
"""

from __future__ import annotations

from zemir.config import RunConfig
from zemir.models.linear import LinearRegressionModel
from zemir.pipeline import run_pipeline


def main() -> None:
    config = RunConfig()
    model = LinearRegressionModel()
    result = run_pipeline(config, model)

    score = result.train_result.validation_score
    print(f"run_id: {result.run_id}")
    print(f"validation mean_corr: {score.mean_corr:.4f}  sharpe: {score.sharpe:.4f}")
    print(f"live predictions: {len(result.live_predictions)} rows")
    for submission in result.submissions:
        print(
            f"submitted to {submission.model_name} ({submission.model_id}): "
            f"submission_id={submission.submission_id}"
        )


if __name__ == "__main__":
    main()
