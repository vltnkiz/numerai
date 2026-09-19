#!/usr/bin/env python3
"""EXPERIMENT entrypoint: run the pipeline and score it WITHOUT submitting.

Same code path and same artifacts as scripts/run_pipeline.py — it just stops
before the upload. There is no submission code here and none in `run_pipeline`,
so this cannot fire a live submission however it is invoked.

Artifacts land in runs/experiments/<run_id>/, apart from the production run
record in runs/<run_id>/.

Usage:
  python scripts/run_experiment.py --strategy zemir_01           # full scale, ~1h
  python scripts/run_experiment.py --strategy zemir_01 --smoke   # seconds
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from zemir.config import LIVE, SMOKE, STRATEGIES, smoke as smoke_strategy
from zemir.data import download
from zemir.pipeline import RUNS_DIR, run_pipeline
from zemir.strategy import build_trainers

EXPERIMENTS_DIR = RUNS_DIR / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="linear")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny, fast run to check the code path — scores are not meaningful",
    )
    args = parser.parse_args()

    config = SMOKE if args.smoke else LIVE
    strategy = STRATEGIES[args.strategy]
    if args.smoke:
        strategy = smoke_strategy(strategy)
    suffix = "-smoke" if args.smoke else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    dataset = download(
        config.data_version, config.feature_set, target_column=config.target_column
    )
    result = run_pipeline(
        config,
        run_id=f"{stamp}-{args.strategy}{suffix}",
        trainers=build_trainers(strategy),
        dataset=dataset,
        blend=strategy.blend,
        runs_dir=EXPERIMENTS_DIR,
    )

    print(f"run_id: {result.run_id}")
    if args.smoke:
        print(f"  SMOKE RUN (max_eras={config.max_eras}) — scores are not meaningful")
    for name, score in result.validation_scores.items():
        print(f"  {name}: mean_corr={score.mean_corr:.4f}  sharpe={score.sharpe:.4f}")
    print(
        f"combined validation mean_corr: {result.combined_validation_score.mean_corr:.4f}  "
        f"sharpe: {result.combined_validation_score.sharpe:.4f}  "
        f"smart_sharpe: {result.combined_validation_score.smart_sharpe:.4f}"
    )
    print(f"live predictions: {len(result.live_predictions)} rows (not submitted)")
    print(f"artifacts: {result.run_dir}")


if __name__ == "__main__":
    main()
