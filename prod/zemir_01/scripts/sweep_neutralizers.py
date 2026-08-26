#!/usr/bin/env python3
"""Sweep neutralizer-set size over a cached fit and print the ranked comparison table.

Issue #35: ranks features by measured exposure of the shipped blend
(linear 0.3 / era_boost 0.7, per #30), then compares neutralizing against only
the top-K most-exposed features against today's status quo — the full
42-feature set — at every K.

Usage:
  python scripts/sweep_neutralizers.py                          # latest cache
  python scripts/sweep_neutralizers.py <run_id>
  python scripts/sweep_neutralizers.py --ks 5 10 15 21 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zemir.config import ENSEMBLE_MODEL_WEIGHTS, neutralizer_subset_sweep
from zemir.harness import (
    HARNESS_DIR,
    PREDICTIONS_FILENAME,
    rank_feature_exposure,
    score_configs,
)

REPORTED_COLUMNS = [
    "eras",
    "mean_corr",
    "sharpe",
    "smart_sharpe",
    "mmc_eras",
    "mean_corr_window",
    "mean_mmc",
    "mmc_sharpe",
    "max_feature_corr",
    "payout",
]


def latest_cache(harness_dir: Path) -> Path:
    caches = [d for d in harness_dir.iterdir() if (d / PREDICTIONS_FILENAME).exists()]
    if not caches:
        raise SystemExit(f"no fitted cache in {harness_dir} — run scripts/fit_harness.py first")
    return max(caches, key=lambda d: d.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?", default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 21, 30])
    args = parser.parse_args()

    run_dir = HARNESS_DIR / args.run_id if args.run_id else latest_cache(HARNESS_DIR)
    models = tuple(ENSEMBLE_MODEL_WEIGHTS)
    weights = tuple(ENSEMBLE_MODEL_WEIGHTS[m] for m in models)

    ranking = rank_feature_exposure(run_dir, models=models, weights=ENSEMBLE_MODEL_WEIGHTS)
    configs = neutralizer_subset_sweep(
        list(ranking.index), tuple(args.ks), models=models, weights=weights
    )
    result = score_configs(configs, run_dir=run_dir)

    ranked = result.summary.sort_values("payout", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"cache: {run_dir}\n")
        print("feature exposure ranking (top 10, mean |corr| across eras):")
        print(ranking.head(10).to_string(float_format=lambda v: f"{v:.6f}"))
        print()
        print(ranked[REPORTED_COLUMNS].to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
