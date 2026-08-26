#!/usr/bin/env python3
"""Sweep model weights over a cached fit and print the ranked comparison table.

The reusable half of #30's deliverable: given any cached fit (from
scripts/fit_harness.py), sweep how much weight each model gets in the blend
and rank by payout. Reused whenever a model joins, leaves, or the fit changes
— not a one-off for linear vs era_boost.

Usage:
  python scripts/sweep_weights.py                                   # latest cache, linear+era_boost, R=10
  python scripts/sweep_weights.py <run_id>
  python scripts/sweep_weights.py --models linear era_boost --resolution 20 --proportion 0.95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zemir.config import weight_sweep
from zemir.harness import HARNESS_DIR, PREDICTIONS_FILENAME, score_configs

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
    parser.add_argument("--models", nargs="+", default=["linear", "era_boost"])
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--proportion", type=float, default=0.95)
    args = parser.parse_args()

    run_dir = HARNESS_DIR / args.run_id if args.run_id else latest_cache(HARNESS_DIR)
    configs = weight_sweep(
        tuple(args.models),
        resolution=args.resolution,
        neutralization_proportion=args.proportion,
    )
    result = score_configs(configs, run_dir=run_dir)

    ranked = result.summary.sort_values("payout", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"cache: {run_dir}\n")
        print(ranked[REPORTED_COLUMNS].to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
