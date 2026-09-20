#!/usr/bin/env python3
"""Sweep blends over a cached fit and print the comparison table.

The cheap half of the measurement backbone: seconds, repeatable, and never
refits. Point it at a run written by fit_harness.py. The table always includes a
`production` row — `PRODUCTION_STRATEGY` scored under the same transform as the
rest — and reports its payout as the baseline, or says plainly that this cache
cannot express it.

Usage:
  python scripts/score_harness.py                 # the most recent cache
  python scripts/score_harness.py <run_id>        # a specific one
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zemir.config import PRODUCTION_STRATEGY, scoring_sweep
from zemir.harness import (
    HARNESS_DIR,
    PREDICTIONS_FILENAME,
    load_fit_record,
    score_configs,
    unscoreable_reason,
)

# The row that scores what production actually ships, under the same transform as every other row.
BASELINE_ROW = "production"

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
    record = load_fit_record(run_dir)
    strategies = scoring_sweep(record.strategy, features=record.feature_set)
    # The baseline is `PRODUCTION_STRATEGY` itself, never a nearby row: a cache that
    # cannot express it says so below rather than substituting one.
    baseline_problem = unscoreable_reason(PRODUCTION_STRATEGY, record)
    if baseline_problem is None:
        strategies[BASELINE_ROW] = PRODUCTION_STRATEGY
    result = score_configs(strategies, run_dir=run_dir)

    ranked = result.summary.sort_values("payout", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"cache: {run_dir}\n")
        print(ranked[REPORTED_COLUMNS].to_string(float_format=lambda v: f"{v:.6f}"))

    if baseline_problem is None:
        baseline = result.summary.loc[BASELINE_ROW]
        unverified = "" if record.verified else " (unverified: this cache predates recorded strategies)"
        print(f"\nbaseline ({BASELINE_ROW}): payout {baseline['payout']:.6f}{unverified}")
    else:
        # Loud on purpose: a baseline that quietly goes missing (or is swapped for a
        # nearby row) is the defect the old string constant was.
        print(f"\nBASELINE UNAVAILABLE: this cache cannot express PRODUCTION_STRATEGY: {baseline_problem}")
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
