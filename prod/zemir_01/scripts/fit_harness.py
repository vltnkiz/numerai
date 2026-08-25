#!/usr/bin/env python3
"""Fit production's models and cache their validation predictions for the harness.

The expensive half of the measurement backbone (~1 hour at full scale). Scoring
sweeps run off the cache this writes, so this runs once per *model* change, not
once per configuration.

Usage:
  python scripts/fit_harness.py            # full scale, ~1h
  python scripts/fit_harness.py --smoke    # seconds; scores meaningless
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from zemir.config import LIVE, MODEL_NAMES, SMOKE
from zemir.harness import fit_validation_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="ensemble")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = SMOKE if args.smoke else LIVE
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{args.model}" + ("-smoke" if args.smoke else "")

    result = fit_validation_predictions(config, run_id=run_id, model=args.model)
    frame = result.predictions
    print(f"run_id: {run_id}")
    print(f"  rows: {len(frame)}  eras: {frame['era'].nunique()}")
    print(f"  models: {[c for c in frame.columns if c not in ('era', 'target')]}")
    print(f"  cache: {result.predictions_path}")


if __name__ == "__main__":
    main()
