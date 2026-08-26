#!/usr/bin/env python3
"""EXPERIMENT entrypoint: run the pipeline and score it WITHOUT submitting.

Same code path and same artifacts as scripts/run_pipeline.py — it just stops
before the upload. There is no submission code here and none in `run_pipeline`,
so this cannot fire a live submission however it is invoked.

Artifacts land in runs/experiments/<run_id>/, apart from the production run
record in runs/<run_id>/.

Usage:
  python scripts/run_experiment.py --model ensemble           # full scale, ~1h
  python scripts/run_experiment.py --model ensemble --smoke   # seconds
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone

from zemir.config import LIVE, MODEL_NAMES, MODEL_WEIGHTS, SMOKE, build_trainers
from zemir.pipeline import RUNS_DIR, run_pipeline

EXPERIMENTS_DIR = RUNS_DIR / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="linear")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny, fast run to check the code path — scores are not meaningful",
    )
    args = parser.parse_args()

    config = replace(SMOKE if args.smoke else LIVE, model_weights=MODEL_WEIGHTS[args.model])
    suffix = "-smoke" if args.smoke else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = run_pipeline(
        config,
        run_id=f"{stamp}-{args.model}{suffix}",
        trainers=build_trainers(args.model, config),
        runs_dir=EXPERIMENTS_DIR,
    )

    print(f"run_id: {result.run_id}")
    if args.smoke:
        print(
            f"  SMOKE RUN (max_eras={config.max_eras}, "
            f"num_iters={config.xgboost['num_iters']}) — scores are not meaningful"
        )
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
