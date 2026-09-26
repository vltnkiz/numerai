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

from zemir.config import LIVE, MIN_VALIDATION_MEAN_CORR, SMOKE, STRATEGIES, smoke as smoke_strategy
from zemir.data import download, load_meta_model, resolve_feature_sets
from zemir.pipeline import (
    RUNS_DIR,
    feature_set_names,
    format_gate,
    format_run_scores,
    record_gate,
    run_columns,
    run_pipeline,
    score_run,
)

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

    feature_sets = resolve_feature_sets(config.data_version, feature_set_names(config, strategy))
    dataset = download(
        config.data_version,
        run_columns(config, strategy, feature_sets),
        target_column=config.target_column,
    )
    result = run_pipeline(
        config,
        run_id=f"{stamp}-{args.strategy}{suffix}",
        strategy=strategy,
        feature_sets=feature_sets,
        dataset=dataset,
        runs_dir=EXPERIMENTS_DIR,
    )

    print(f"run_id: {result.run_id}")
    if args.smoke:
        print(f"  SMOKE RUN (max_eras={config.max_eras}) — scores are not meaningful")
    # The live gate's own record and verdict, reported rather than enforced:
    # an experiment never submits, so it has nothing to stop.
    passed = record_gate(result, threshold=MIN_VALIDATION_MEAN_CORR)["passed"]
    print(format_gate(result, threshold=MIN_VALIDATION_MEAN_CORR))
    print(f"  the live run {'would submit' if passed else 'would NOT submit'} this")
    print(f"live predictions: {len(result.live_predictions)} rows (not submitted)")
    # The cached meta model, never a refresh: only the live run keeps the
    # shared copy current (issue #101). No try/except either: with no
    # submission to protect, a scoring failure here should just crash.
    scores = score_run(
        result,
        meta_model=load_meta_model(config.data_version),
        scoring_universe=feature_sets[config.feature_set],
    )
    print(format_run_scores(scores))
    print(f"artifacts: {result.run_dir}")


if __name__ == "__main__":
    main()
