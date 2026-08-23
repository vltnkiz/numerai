#!/usr/bin/env python3
"""CLI entrypoint: run one end-to-end zemir pipeline invocation.

Usage:
  python scripts/run_pipeline.py --model linear
  python scripts/run_pipeline.py --model era_boost
  python scripts/run_pipeline.py --model ensemble [--ensemble-weights linear=0.3,era_boost=0.7]
"""

from __future__ import annotations

import argparse

from zemir.config import EnsembleConfig, EraBoostConfig, RunConfig
from zemir.models.base import Model
from zemir.models.era_boost import EraBoostModel
from zemir.models.linear import LinearRegressionModel
from zemir.pipeline import run_pipeline


def _parse_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        if not name or not value:
            raise ValueError(f"malformed --ensemble-weights entry: {pair!r}")
        weights[name] = float(value)
    return weights


def _build_config_and_models(
    model_name: str, ensemble_weights: str | None
) -> tuple[RunConfig, dict[str, Model]]:
    if model_name == "linear":
        return RunConfig(), {"linear": LinearRegressionModel()}
    if model_name == "era_boost":
        era_boost = EraBoostConfig()
        return RunConfig(era_boost=era_boost), {"era_boost": EraBoostModel(era_boost)}
    if model_name == "ensemble":
        era_boost = EraBoostConfig()
        ensemble = (
            EnsembleConfig(weights=_parse_weights(ensemble_weights))
            if ensemble_weights
            else EnsembleConfig()
        )
        config = RunConfig(era_boost=era_boost, ensemble=ensemble)
        models: dict[str, Model] = {
            "linear": LinearRegressionModel(),
            "era_boost": EraBoostModel(era_boost),
        }
        return config, models
    raise ValueError(f"unknown model {model_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "era_boost", "ensemble"], default="linear")
    parser.add_argument(
        "--ensemble-weights",
        default=None,
        help="comma-separated name=weight pairs, e.g. linear=0.3,era_boost=0.7 "
        "(only used with --model ensemble; defaults to EnsembleConfig's default)",
    )
    args = parser.parse_args()

    config, models = _build_config_and_models(args.model, args.ensemble_weights)
    result = run_pipeline(config, models)

    print(f"run_id: {result.run_id}")
    for name, train_result in result.train_results.items():
        score = train_result.validation_score
        print(f"  {name}: mean_corr={score.mean_corr:.4f}  sharpe={score.sharpe:.4f}")
    print(
        f"combined validation mean_corr: {result.validation_score.mean_corr:.4f}  "
        f"sharpe: {result.validation_score.sharpe:.4f}"
    )
    print(f"live predictions: {len(result.live_predictions)} rows")
    for submission in result.submissions:
        print(
            f"submitted to {submission.model_name} ({submission.model_id}): "
            f"submission_id={submission.submission_id}"
        )


if __name__ == "__main__":
    main()
