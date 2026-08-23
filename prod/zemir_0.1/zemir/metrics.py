"""Era-wise correlation scoring, shared across model types.

Ports era-boosting/code/era_boosting.ipynb's era_spearman_corr /
autocorr_penalty / smart_sharpe helpers — the per-era Spearman correlation
against a validation set is the score decision #1 (destination-scoping
grilling) requires before predictions are allowed to reach the submission
stage.
"""

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
