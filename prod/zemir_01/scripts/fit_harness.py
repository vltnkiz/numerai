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
  python scripts/fit_harness.py --feature-set all --strategy era_boost \
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
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from zemir.config import LIVE, SMOKE, STRATEGIES, smoke as smoke_strategy
from zemir.data import load_split, resolve_feature_sets
from zemir.harness import fit_validation_predictions
from zemir.strategy import Strategy, union_columns


def _with_xgboost_overrides(strategy: Strategy, overrides: Mapping[str, object]) -> Strategy:
    """`strategy`, with `overrides` merged onto every model whose trainer is `xgboost`."""
    models = tuple(
        replace(m, params={**m.params, **overrides}) if m.trainer == "xgboost" else m
        for m in strategy.models
    )
    return replace(strategy, models=models)


def _with_linear_trainer(strategy: Strategy, trainer: str, params: Mapping[str, object]) -> Strategy:
    """`strategy`, with the `linear` spec's trainer swapped for `trainer` (issue #38/#50).

    A `ModelSpec` names its own trainer, so `--linear ols|ridge|sgd` is one
    string substitution on the spec rather than a `build_trainers` wrapper
    closing over a lambda.
    """
    models = tuple(
        replace(m, trainer=trainer, params=params) if m.name == "linear" else m
        for m in strategy.models
    )
    return replace(strategy, models=models)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="zemir_01")
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

    strategy = STRATEGIES[args.strategy]
    if args.smoke:
        strategy = smoke_strategy(strategy)
    if args.feature_set is not None:
        # A width-comparison knob: every model at one width, not just the run's
        # default universe — a spec names its own `features` now.
        strategy = replace(
            strategy,
            models=tuple(replace(m, features=args.feature_set) for m in strategy.models),
        )

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
        strategy = _with_xgboost_overrides(strategy, xgb_overrides)

    if args.linear == "ridge":
        strategy = _with_linear_trainer(strategy, "ridge", {"alpha": args.ridge_alpha})
    elif args.linear == "sgd":
        # defaults mirror train_sgd's own signature defaults (sklearn's untuned values)
        strategy = _with_linear_trainer(
            strategy,
            "sgd",
            {
                "alpha": args.sgd_alpha if args.sgd_alpha is not None else 0.0001,
                "eta0": args.sgd_eta0 if args.sgd_eta0 is not None else 0.01,
            },
        )

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
    run_id = f"{stamp}-{args.strategy}{suffix}"

    # Narrowed before either split is read off disk (issue #45's dropped
    # columns), not after — see fit_validation_predictions' docstring.
    feature_sets = resolve_feature_sets(
        config.data_version, strategy.feature_set_names, drop=drop_features or frozenset()
    )
    columns = list(union_columns(strategy, feature_sets))

    result = fit_validation_predictions(
        config,
        strategy,
        run_id=run_id,
        feature_sets=feature_sets,
        load_train=partial(
            load_split,
            config.data_version,
            config.feature_set,
            "train",
            feature_names=columns,
            target_column=config.target_column,
        ),
        load_validation=partial(
            load_split,
            config.data_version,
            config.feature_set,
            "validation",
            feature_names=columns,
            target_column=config.target_column,
        ),
    )
    frame = result.predictions
    print(f"run_id: {run_id}")
    print(f"  rows: {len(frame)}  eras: {frame['era'].nunique()}")
    print(f"  models: {[c for c in frame.columns if c not in ('era', 'target')]}")
    print(f"  cache: {result.predictions_path}")


if __name__ == "__main__":
    main()
