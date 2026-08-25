#!/usr/bin/env python3
"""Sweep scoring configurations over a cached fit and print the comparison table.

The cheap half of the measurement backbone: seconds, repeatable, and never
refits. Point it at a run written by fit_harness.py.

Usage:
  python scripts/score_harness.py                 # the most recent cache
  python scripts/score_harness.py <run_id>        # a specific one
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zemir.config import PRODUCTION_BASELINE, scoring_sweep
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
    args = parser.parse_args()

    run_dir = HARNESS_DIR / args.run_id if args.run_id else latest_cache(HARNESS_DIR)
    result = score_configs(scoring_sweep(), run_dir=run_dir)

    ranked = result.summary.sort_values("payout", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"cache: {run_dir}\n")
        print(ranked[REPORTED_COLUMNS].to_string(float_format=lambda v: f"{v:.6f}"))

    baseline = result.summary.loc[PRODUCTION_BASELINE]
    print(f"\nbaseline ({PRODUCTION_BASELINE}): payout {baseline['payout']:.6f}")
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
