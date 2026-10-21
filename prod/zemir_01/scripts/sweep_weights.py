#!/usr/bin/env python3
"""Sweep model weights over a cached fit and print the ranked comparison table.

The reusable half of #30's deliverable: given any cached fit (from
scripts/fit_harness.py), sweep how much weight each model gets in the blend
and rank by the validation payout proxy, beside production's own row. Reused whenever a model joins, leaves, or the fit changes
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
from zemir.harness import (
    HARNESS_DIR,
    PREDICTIONS_FILENAME,
    load_fit_record,
    ranked_table,
    score_configs,
    with_baseline,
)


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
    record = load_fit_record(run_dir)
    strategies = weight_sweep(
        record.strategy,
        tuple(args.models),
        features=record.feature_set,
        resolution=args.resolution,
        neutralization_proportion=args.proportion,
    )
    strategies, baseline_problem = with_baseline(strategies, record)
    result = score_configs(strategies, run_dir=run_dir)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"cache: {run_dir}\n")
        print(ranked_table(result).to_string(float_format=lambda v: f"{v:.6f}"))
    if baseline_problem is not None:
        print(f"\nBASELINE UNAVAILABLE: this cache cannot express PRODUCTION_STRATEGY: {baseline_problem}")
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
