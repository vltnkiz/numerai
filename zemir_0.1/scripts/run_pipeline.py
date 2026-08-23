#!/usr/bin/env python3
"""CLI entrypoint: run one end-to-end zemir pipeline invocation.

Usage: python scripts/run_pipeline.py [--model linear|era_boost]
"""

from __future__ import annotations

import argparse

from zemir.config import EraBoostConfig, RunConfig
from zemir.models.base import Model
from zemir.models.era_boost import EraBoostModel
from zemir.models.linear import LinearRegressionModel
from zemir.pipeline import run_pipeline


def _build_config_and_model(model_name: str) -> tuple[RunConfig, Model]:
    if model_name == "linear":
        return RunConfig(), LinearRegressionModel()
    if model_name == "era_boost":
        era_boost = EraBoostConfig()
        return RunConfig(era_boost=era_boost), EraBoostModel(era_boost)
    raise ValueError(f"unknown model {model_name!r}; expected 'linear' or 'era_boost'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "era_boost"], default="linear")
    args = parser.parse_args()

    config, model = _build_config_and_model(args.model)
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
