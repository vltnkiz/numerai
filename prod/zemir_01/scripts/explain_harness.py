#!/usr/bin/env python3
"""Print what each cached model leans on, without refitting anything.

The other cheap half of the measurement backbone, beside score_harness.py: the
models fit_harness.py fitted are kept next to the predictions, so a question
about them costs seconds rather than the ~1 h fit. Point it at a run written by
fit_harness.py.

Gain / weight scale only — no SHAP. A linear model prints its coefficients
(`coef`); a booster prints `weight` (splits on the feature), `gain` and `cover`.

Usage:
  python scripts/explain_harness.py                 # the most recent cache
  python scripts/explain_harness.py <run_id>        # a specific one
  python scripts/explain_harness.py --top 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zemir.harness import HARNESS_DIR, PREDICTIONS_FILENAME, NoPersistedModels, explain_models

CONTRACT = "describes these fits exactly; not an estimate of what matters in the data."


def latest_cache(harness_dir: Path) -> Path:
    caches = [d for d in harness_dir.iterdir() if (d / PREDICTIONS_FILENAME).exists()]
    if not caches:
        raise SystemExit(f"no fitted cache in {harness_dir} — run scripts/fit_harness.py first")
    return max(caches, key=lambda d: d.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?", default=None)
    parser.add_argument("--top", type=int, default=20, help="features to show per model")
    args = parser.parse_args()

    run_dir = HARNESS_DIR / args.run_id if args.run_id else latest_cache(HARNESS_DIR)
    try:
        explained = explain_models(run_dir)
    except NoPersistedModels as error:
        raise SystemExit(str(error)) from None

    print(f"cache: {run_dir}")
    print(f"Explanation {CONTRACT}\n")
    for name, table in explained.items():
        if "gain" in table:
            split_on = int((table["weight"] > 0).sum())
            print(f"{name}: booster split on {split_on} of {len(table)} features, by gain")
        else:
            print(f"{name}: linear, {len(table)} coefficients, by |coef|")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(table.head(args.top).to_string(float_format=lambda v: f"{v:.6g}"))
        print()


if __name__ == "__main__":
    main()
