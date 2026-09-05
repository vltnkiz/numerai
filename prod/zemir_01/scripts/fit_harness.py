#!/usr/bin/env python3
"""Fit production's models and cache their validation predictions for the harness.

The expensive half of the measurement backbone (~1 hour at full scale). Scoring
sweeps run off the cache this writes, so this runs once per *model* change, not
once per configuration.

Usage:
  python scripts/fit_harness.py                     # full scale, ~1h
  python scripts/fit_harness.py --smoke              # seconds; scores meaningless
  python scripts/fit_harness.py --feature-set medium --colsample-bytree 0.005
  python scripts/fit_harness.py --feature-set all --linear ridge --ridge-alpha 10
  python scripts/fit_harness.py --feature-set all --linear ridge --ridge-alpha 10 \
      --drop-columns-file ~/all_width_drop_columns.json
  python scripts/fit_harness.py --feature-set all --linear sgd \
      --drop-columns-file ~/all_width_drop_columns.json
  python scripts/fit_harness.py --feature-set all --linear sgd \
      --sgd-alpha 0.3 --sgd-eta0 0.0003 \
      --drop-columns-file ~/all_width_drop_columns.json
  python scripts/fit_harness.py --feature-set all --model era_boost \
      --max-eras 120 --colsample-bytree 0.00123 --trees-per-step 200 \
      --learning-rate 0.005 --drop-columns-file ~/all_width_drop_columns.json

`--feature-set`/`--colsample-bytree`/`--linear`/`--ridge-alpha` are issue #38's
width-comparison knobs, layered on top of `LIVE`/`SMOKE` without changing their
defaults — `LIVE` still ships `small`/OLS until the width decision lands in
`zemir/config.py`. `--drop-columns-file` is issue #45's: a JSON file with a
`"drop"` key listing feature names to exclude from every trainer and from
prediction (zero-variance/near-duplicate columns at `all` width — see
scratch/find_dead_columns.py-style scans, not committed here since the list
is data-derived, not code). `--linear sgd` is issue #50's chunked-`partial_fit`
escalation past Ridge's memory ceiling at `all` width — see `train_sgd`'s
docstring for what it trades away and why. `--sgd-alpha`/`--sgd-eta0` are
issue #51's tuned hyperparameters for it (sklearn's untuned defaults are
unstable at `all` width — see `train_sgd`'s docstring); they default to
`train_sgd`'s own defaults (sklearn's untuned values) when omitted, same as
leaving `--linear sgd` off the earlier examples did before #51.

`--num-iters`/`--trees-per-step`/`--max-depth`/`--learning-rate` are issue
#54's era_boost tuning knobs, the tree-stage counterparts to `--sgd-alpha`/
`--sgd-eta0` — they default to `train_xgboost`'s own signature defaults
(today's shipped values) when omitted. `--max-eras` generalizes `--smoke`'s
hardcoded `max_eras=40`/`num_iters=2` into an arbitrary era-count override
that leaves every other hyperparameter alone — `--smoke` still exists
unchanged for a seconds-scale code-path check; `--max-eras` is for a
comparative tuning proxy (issue #54) sized for a trustworthy relative
ranking, not merely "does it run."
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from zemir.config import LIVE, MODEL_NAMES, PipelineConfig, SMOKE, build_trainers
from zemir.harness import fit_validation_predictions
from zemir.models import train_ridge, train_sgd


def _build_trainers_with_ridge(model: str, config: PipelineConfig, *, alpha: float) -> dict:
    """`build_trainers`, with the `linear` trainer (if present) swapped for Ridge."""
    trainers = build_trainers(model, config)
    if "linear" in trainers:
        trainers["linear"] = lambda X, y, era: train_ridge(X, y, alpha=alpha)
    return trainers


def _build_trainers_with_sgd(
    model: str, config: PipelineConfig, *, alpha: float, eta0: float
) -> dict:
    """`build_trainers`, with the `linear` trainer (if present) swapped for chunked SGD (issue #50)."""
    trainers = build_trainers(model, config)
    if "linear" in trainers:
        trainers["linear"] = lambda X, y, era: train_sgd(X, y, era, alpha=alpha, eta0=eta0)
    return trainers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="ensemble")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--feature-set", choices=["small", "medium", "all"], default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--linear", choices=["ols", "ridge", "sgd"], default="ols")
    parser.add_argument("--ridge-alpha", type=float, default=None)
    parser.add_argument("--sgd-alpha", type=float, default=None)
    parser.add_argument("--sgd-eta0", type=float, default=None)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--trees-per-step", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-eras", type=int, default=None)
    parser.add_argument("--drop-columns-file", type=Path, default=None)
    args = parser.parse_args()

    if args.linear == "ridge" and args.ridge_alpha is None:
        parser.error("--linear ridge requires --ridge-alpha")

    drop_features = None
    if args.drop_columns_file is not None:
        drop_features = frozenset(
            json.loads(args.drop_columns_file.expanduser().read_text())["drop"]
        )

    config = SMOKE if args.smoke else LIVE
    if args.feature_set is not None:
        config = replace(config, feature_set=args.feature_set)
    if args.max_eras is not None:
        config = replace(config, max_eras=args.max_eras)

    xgb_overrides = {}
    if args.colsample_bytree is not None:
        xgb_overrides["colsample_bytree"] = args.colsample_bytree
    if args.num_iters is not None:
        xgb_overrides["num_iters"] = args.num_iters
    if args.trees_per_step is not None:
        xgb_overrides["trees_per_step"] = args.trees_per_step
    if args.max_depth is not None:
        xgb_overrides["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        xgb_overrides["learning_rate"] = args.learning_rate
    if xgb_overrides:
        config = replace(config, xgboost={**config.xgboost, **xgb_overrides})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-smoke" if args.smoke else ""
    if args.feature_set is not None:
        suffix += f"-{args.feature_set}"
    if args.linear == "ridge":
        suffix += f"-ridge{args.ridge_alpha:g}"
    if args.linear == "sgd":
        suffix += "-sgd"
        if args.sgd_alpha is not None:
            suffix += f"-a{args.sgd_alpha:g}"
        if args.sgd_eta0 is not None:
            suffix += f"-e{args.sgd_eta0:g}"
    if args.max_eras is not None:
        suffix += f"-eras{args.max_eras}"
    if args.trees_per_step is not None:
        suffix += f"-tps{args.trees_per_step}"
    if args.num_iters is not None:
        suffix += f"-iters{args.num_iters}"
    if args.max_depth is not None:
        suffix += f"-depth{args.max_depth}"
    if args.learning_rate is not None:
        suffix += f"-lr{args.learning_rate:g}"
    if drop_features:
        suffix += f"-drop{len(drop_features)}"
    run_id = f"{stamp}-{args.model}{suffix}"

    trainers_override = None
    if args.linear == "ridge":
        trainers_override = partial(_build_trainers_with_ridge, alpha=args.ridge_alpha)
    elif args.linear == "sgd":
        # defaults mirror train_sgd's own signature defaults (sklearn's untuned values)
        trainers_override = partial(
            _build_trainers_with_sgd,
            alpha=args.sgd_alpha if args.sgd_alpha is not None else 0.0001,
            eta0=args.sgd_eta0 if args.sgd_eta0 is not None else 0.01,
        )

    result = fit_validation_predictions(
        config,
        run_id=run_id,
        model=args.model,
        build_trainers=trainers_override or build_trainers,
        drop_features=drop_features,
    )
    frame = result.predictions
    print(f"run_id: {run_id}")
    print(f"  rows: {len(frame)}  eras: {frame['era'].nunique()}")
    print(f"  models: {[c for c in frame.columns if c not in ('era', 'target')]}")
    print(f"  cache: {result.predictions_path}")


if __name__ == "__main__":
    main()
