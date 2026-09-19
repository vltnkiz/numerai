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

`zemir.harness.fit_validation_predictions` loads its splits one at a time for
the same reasons and stays that way: it hands this module a train loader, and
loads validation only after this returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

from zemir.data import scoring_window
from zemir.strategy import (
    FeatureSets,
    ModelSpec,
    Strategy,
    build_trainer,
    model_columns,
    union_columns,
)


@dataclass(frozen=True)
class FittedSpec:
    """A fitted model, bound to the columns it was fitted on.

    Predicting on the wrong columns is silent — a tree stage indexes by
    position — so the columns travel with the model, and `predict` takes a
    frame of *any* width that contains them.
    """

    spec: ModelSpec
    columns: tuple[str, ...]
    model: object

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[list(self.columns)])


@dataclass
class StrategyFit:
    models: dict[str, FittedSpec]  # keyed by `ModelSpec.name`, in declared order
    train_eras: int


def fit_strategy(
    strategy: Strategy,
    load_train: Callable[[], pd.DataFrame],
    *,
    feature_sets: FeatureSets,
    max_eras: int | None = None,
) -> StrategyFit:
    """Fit every model in `strategy` on the split `load_train` returns.

    `load_train` must return the train split *including* every column in the
    strategy's union (see `zemir.strategy.union_columns`) plus `target` and
    `era`; the target-bearing, last-`max_eras` window is taken here. See the
    module docstring for what this owns and why.
    """
    train_df = scoring_window(load_train(), max_eras)
    train_eras = int(train_df["era"].nunique())

    union = union_columns(strategy, feature_sets)
    X, y, era = train_df[list(union)], train_df["target"], train_df["era"]
    del train_df

    columns = {spec.name: model_columns(spec, feature_sets) for spec in strategy.models}
    fit_order = sorted(strategy.models, key=lambda spec: len(columns[spec.name]))

    fitted: dict[str, FittedSpec] = {}
    for position, spec in enumerate(fit_order):
        pa.default_memory_pool().release_unused()
        spec_columns = columns[spec.name]
        X_spec = X if spec_columns == union else X[list(spec_columns)]
        if position == len(fit_order) - 1:
            X = None  # the last model needs nothing but its own slice
        fitted[spec.name] = FittedSpec(
            spec=spec, columns=spec_columns, model=build_trainer(spec)(X_spec, y, era)
        )
        X_spec = None

    X = y = era = None
    pa.default_memory_pool().release_unused()
    return StrategyFit(
        models={spec.name: fitted[spec.name] for spec in strategy.models},
        train_eras=train_eras,
    )
