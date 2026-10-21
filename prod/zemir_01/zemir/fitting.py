"""Fit a Strategy's models, each on its own feature slice.

This module owns the memory discipline of a fit. Every rule below is
load-bearing at a **43.2 GiB measured peak** against `MIN_AVAILABLE_MEMORY_GIB`'s
46 GiB floor (`zemir.config`), and none of them is visible from a call site —
so they live here, in one place, rather than as obligations on each caller:

1. **One frame live, and the caller cannot hold it.** `fit_strategy` takes a
   *loader*, not a frame. It is therefore the only holder of the train split
   from the moment it is read until the last model is fitted; a caller that
   kept its own `train_df` binding was the ~8.7 GiB gap issue #45's
   investigation stalled on (#47).
2. **Slice once, then drop the wide frame.** The union of every model's columns
   is cut out of the train frame once, and the frame itself is dropped before
   any trainer runs.
3. **A model naming the whole union is handed the union itself**, not a copy.
   So a strategy whose models all name one feature set — every shipped one —
   holds exactly the one full-width matrix it always did. Only a strictly
   narrower model costs a slice, and that slice is freed the moment its model
   is fitted. The union is freed as soon as the last model has taken its slice.
4. **Narrowest first, widest last.** A trainer's transient scales with its
   width (OLS's float64 upcast is ~16 bytes per cell), and the union stays live
   under every model but the last — so the widest model runs last, with
   nothing else beside it, and a narrow model never sits under a wide one's
   transient. Results are order-independent (every trainer is seeded), and the
   returned models are still in declared order.
5. **`release_unused()` before *each* trainer, and once after the last.**
   pyarrow's default pool (mimalloc) keeps freed pages for reuse rather than
   returning them to the OS, so the parquet read's arena stays resident through
   every trainer in turn. Measured at `all` width: 20.11 GiB resident right
   after the load, 9.77 GiB after this call — **10.3 GiB** that each trainer
   was otherwise fitting on top of, and the difference between issue #52's
   `all`-width fit breaching its memory guard and clearing it. The release
   *after* the last trainer frees the fit's own residue before the caller
   loads anything else (~11 GiB before the harness's validation load, #51).

6. **Rows are shared the way columns are.** Each model fits on the rows its
   own `target` carries, less that target's last `fit_purge_eras` eras. When
   every model names one target — every shipped strategy — they share one row
   set and are handed the union as above; only a model whose rows differ from
   the rest costs a copy.

`zemir.harness.fit_validation_predictions` loads its splits one at a time for
the same reasons and stays that way: it hands this module a train loader, and
loads validation only after this returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

from zemir.data import last_eras
from zemir.models import FittedModel
from zemir.strategy import (
    FeatureSets,
    ModelSpec,
    Strategy,
    build_trainer,
    fit_purge_eras,
    model_columns,
    union_columns,
)


@dataclass
class StrategyFit:
    models: dict[str, FittedModel]  # keyed by `ModelSpec.name`, in declared order
    train_eras: dict[str, int]  # keyed the same way: how many eras each model saw


def fit_rows(train_df: pd.DataFrame, target: str, max_eras: int | None) -> pd.Index:
    """The train rows a model fitted on `target` uses.

    The rows that carry `target`, less the last `fit_purge_eras(target)` eras,
    whose returns overlap validation's first eras; then, for a smoke run, the
    last `max_eras` of what is left.
    """
    bearing = train_df.loc[train_df[target].notna(), ["era"]]
    eras = sorted(bearing["era"].unique(), key=int)
    purge = fit_purge_eras(target)
    if len(eras) <= purge:
        raise ValueError(
            f"no train eras left for {target!r}: it has {len(eras)}, and its horizon drops the last {purge}"
        )
    kept = bearing[bearing["era"].isin(eras[: len(eras) - purge])]
    if max_eras is not None:
        kept = last_eras(kept, max_eras)
    return kept.index


def _checked(fitted: object, spec: ModelSpec, columns: tuple[str, ...]) -> FittedModel:
    """`fitted`, once it is known to be bound to exactly the columns `spec` was handed.

    A `FittedModel` carries its own columns, taken by the trainer from the frame
    it was handed — so this is the one place the two are compared, and it fails
    loudly: a model bound to other columns would predict on the wrong ones
    without complaint.
    """
    if not isinstance(fitted, FittedModel):
        raise TypeError(
            f"trainer {spec.trainer!r} for model {spec.name!r} returned "
            f"{type(fitted).__name__}, not a FittedModel"
        )
    if fitted.columns != columns:
        raise ValueError(
            f"model {spec.name!r} was fitted on {len(fitted.columns)} columns that are not the "
            f"{len(columns)} its feature set {spec.features!r} names"
        )
    return fitted


def fit_strategy(
    strategy: Strategy,
    load_train: Callable[[], pd.DataFrame],
    *,
    feature_sets: FeatureSets,
    max_eras: int | None = None,
) -> StrategyFit:
    """Fit every model in `strategy` on the split `load_train` returns.

    `load_train` must return the train split *including* every column in the
    strategy's union (see `zemir.strategy.union_columns`), `era`, and every
    model's `target`; each model's rows are taken here (`fit_rows`). See the
    module docstring for what this owns and why.
    """
    train_df = load_train()
    targets = list(dict.fromkeys(spec.target for spec in strategy.models))
    rows = {target: fit_rows(train_df, target, max_eras) for target in targets}
    shared = rows[targets[0]]
    for target in targets[1:]:
        shared = shared.union(rows[target], sort=False)

    union = union_columns(strategy, feature_sets)
    X = train_df.loc[shared, list(union)]
    y = train_df.loc[shared, targets]
    era = train_df.loc[shared, "era"]
    del train_df
    eras_seen = {target: int(era.loc[rows[target]].nunique()) for target in targets}

    columns = {spec.name: model_columns(spec, feature_sets) for spec in strategy.models}
    fit_order = sorted(strategy.models, key=lambda spec: len(columns[spec.name]))

    fitted: dict[str, FittedModel] = {}
    for position, spec in enumerate(fit_order):
        pa.default_memory_pool().release_unused()
        spec_columns = columns[spec.name]
        if len(rows[spec.target]) == len(shared):
            X_spec = X if spec_columns == union else X[list(spec_columns)]
            y_spec, era_spec = y[spec.target], era
        else:
            spec_rows = rows[spec.target]
            X_spec = X.loc[spec_rows, list(spec_columns)]
            y_spec, era_spec = y.loc[spec_rows, spec.target], era.loc[spec_rows]
        if position == len(fit_order) - 1:
            X = None  # the last model needs nothing but its own slice
        fitted[spec.name] = _checked(
            build_trainer(spec)(X_spec, y_spec, era_spec), spec, spec_columns
        )
        X_spec = y_spec = era_spec = None

    X = y = era = None
    pa.default_memory_pool().release_unused()
    return StrategyFit(
        models={spec.name: fitted[spec.name] for spec in strategy.models},
        train_eras={spec.name: eras_seen[spec.target] for spec in strategy.models},
    )
