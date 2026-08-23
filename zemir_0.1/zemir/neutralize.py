"""Feature neutralization stage: reduce predictions' linear exposure to given features.

Ports feature-neutralization/code/feature_neutralization.ipynb's neutralize()
unchanged. Applied to a model's raw predictions (zemir/models/base.py's
Model.predict() is deliberately unneutralized) before they reach the
submission stage, per decision #1 (destination-scoping grilling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(
    df: pd.DataFrame,
    columns: list[str],
    neutralizers: list[str],
    proportion: float,
) -> np.ndarray:
    """Remove `proportion` of the linear component of `columns` explained by `neutralizers`.

    Exposures are regressed with a bias term via `np.linalg.pinv`. Ported
    unchanged from the notebook; `df` is expected to already be scoped to
    rows an exposure fit should be computed over (an era, in practice).
    """
    scores = df[columns].values
    exposures = df[neutralizers].values
    exposures = np.hstack((exposures, np.ones((len(exposures), 1))))
    correction = proportion * (exposures @ (np.linalg.pinv(exposures) @ scores))
    return scores - correction


def neutralize_predictions(
    df: pd.DataFrame,
    neutralizers: list[str],
    proportion: float,
    *,
    prediction_col: str = "prediction",
    era_col: str = "era",
) -> pd.Series:
    """Neutralize `prediction_col` against `neutralizers`, era by era.

    The stage entrypoint: exposures are era-specific, so `neutralize` is
    applied per era rather than across the whole frame at once, matching the
    notebook's `groupby("era").apply(neutralize, ...)` usage.

    Grouped manually (rather than via `DataFrame.groupby().apply()`) because
    pandas collapses a single-group `apply()` result into a DataFrame instead
    of a Series — live data is always a single era (`"X"`), so this path is
    exercised on every live-round run, not just multi-era training data.
    """
    parts = [
        pd.Series(
            neutralize(era_df, [prediction_col], neutralizers, proportion).ravel(),
            index=era_df.index,
        )
        for _, era_df in df.groupby(era_col, group_keys=False)
    ]
    neutralized = pd.concat(parts).loc[df.index]
    return neutralized.rename(f"{prediction_col}_neutralized")
