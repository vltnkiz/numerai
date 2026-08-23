from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def era_corr(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    prediction_col: str = "prediction",
    era_col: str = "era",
) -> pd.Series:
    corrs = df.groupby(era_col).apply(
        lambda d: spearmanr(d[prediction_col], d[target_col])[0]
    )
    return corrs.loc[sorted(corrs.index, key=int)]


def _autocorr_penalty(x: np.ndarray) -> float:
    n = len(x)
    p = np.corrcoef(x[:-1], x[1:])[0, 1]
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p**i for i in range(1, n)]))


@dataclass
class ValidationScore:
    era_corr: pd.Series
    mean_corr: float
    std_corr: float
    sharpe: float
    smart_sharpe: float


def score_validation(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    prediction_col: str = "prediction",
    era_col: str = "era",
) -> ValidationScore:
    corrs = era_corr(
        df, target_col=target_col, prediction_col=prediction_col, era_col=era_col
    )
    penalty = _autocorr_penalty(corrs.values.astype(float))
    return ValidationScore(
        era_corr=corrs,
        mean_corr=float(corrs.mean()),
        std_corr=float(corrs.std()),
        sharpe=float(corrs.mean() / corrs.std()),
        smart_sharpe=float(corrs.mean() / (corrs.std(ddof=1) * penalty)),
    )


def neutralize(
    df: pd.DataFrame,
    columns: list[str],
    neutralizers: list[str],
    proportion: float,
) -> np.ndarray:
    """Remove `proportion` of the linear component of `columns` explained by `neutralizers`."""
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
    # Grouped manually rather than via groupby().apply(): pandas collapses a
    # single-group apply() result into a DataFrame instead of a Series, and
    # live data is always a single era.
    parts = [
        pd.Series(
            neutralize(era_df, [prediction_col], neutralizers, proportion).ravel(),
            index=era_df.index,
        )
        for _, era_df in df.groupby(era_col, group_keys=False)
    ]
    neutralized = pd.concat(parts).loc[df.index]
    return neutralized.rename(f"{prediction_col}_neutralized")


def rank_normalize(predictions: pd.Series) -> pd.Series:
    """Rescale to (0, 1) exclusive — Numerai rejects submissions outside that range."""
    n = len(predictions)
    ranks = predictions.rank(method="average")
    return ((ranks - 0.5) / n).rename(predictions.name)


class PredictionSanityError(RuntimeError):
    pass


def validate_predictions(predictions: pd.Series, era: pd.Series) -> None:
    if predictions.isna().any():
        n = int(predictions.isna().sum())
        raise PredictionSanityError(f"{n} null prediction(s) found — refusing to submit")

    if ((predictions <= 0) | (predictions >= 1)).any():
        raise PredictionSanityError("prediction(s) outside (0, 1) — refusing to submit")

    variance_by_era = predictions.groupby(era).var()
    constant_eras = variance_by_era[variance_by_era.fillna(0) == 0].index.tolist()
    if constant_eras:
        raise PredictionSanityError(
            f"zero-variance predictions in era(s) {constant_eras} — refusing to submit"
        )
