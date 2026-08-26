from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr
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


def project(scores: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """The linear component of `scores` explained by `exposures` (plus an intercept).

    Solved with `lstsq` rather than by forming `pinv(exposures)`: both take the
    minimum-norm least-squares solution and agree to ~1e-15, but lstsq never
    materializes the pseudo-inverse and is roughly twice as fast — which matters
    because a sweep solves this once per era.
    """
    exposures = np.hstack((exposures, np.ones((len(exposures), 1))))
    return exposures @ np.linalg.lstsq(exposures, scores, rcond=None)[0]


def neutralize(
    df: pd.DataFrame,
    columns: list[str],
    neutralizers: list[str],
    proportion: float,
) -> np.ndarray:
    """Remove `proportion` of the linear component of `columns` explained by `neutralizers`."""
    scores = df[columns].values
    return scores - proportion * project(scores, df[neutralizers].values)


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


# ---------------------------------------------------------------------------
# Numerai's *paid* metrics.
#
# `score_validation` above uses plain Spearman, which is not what Numerai pays
# on. Payout is `0.75 * corr20 + 2.25 * mmc20` — MMC carries three times the
# weight of CORR — so a comparison made on Spearman alone can pick the wrong
# architecture. These wrap `numerai-tools`, Numerai's own reference
# implementation, rather than reimplementing the formulas.
#
# All three are invariant to any strictly monotone transform of the
# predictions, `rank_normalize` included: they rank internally. Verified
# exactly (delta 0.00e+00) — see issue #26.
# ---------------------------------------------------------------------------

PAYOUT_CORR_MULTIPLIER = 0.75
PAYOUT_MMC_MULTIPLIER = 2.25


def _by_era(era: pd.Series, score: Callable[[pd.Index], pd.Series]) -> pd.DataFrame:
    """Apply a whole-column scorer to each era; rows are eras, columns predictions."""
    scores = {e: score(idx) for e, idx in era.groupby(era).groups.items()}
    frame = pd.DataFrame(scores).T
    frame.index.name = "era"
    return frame.loc[sorted(frame.index, key=int)]


def era_numerai_corr(
    predictions: pd.DataFrame, target: pd.Series, era: pd.Series
) -> pd.DataFrame:
    """Numerai's paid CORR per era: rank -> gaussianize -> ^1.5 -> Pearson."""
    return _by_era(
        era, lambda idx: numerai_corr(predictions.loc[idx], target.loc[idx])
    )


def era_mmc(
    predictions: pd.DataFrame,
    target: pd.Series,
    era: pd.Series,
    meta_model: pd.Series,
) -> pd.DataFrame:
    """MMC per era, over the eras the meta model covers.

    The meta model spans a window of validation (96 eras in v5.0), not all of
    it, so this returns fewer rows than `era_numerai_corr` — deliberately.
    Comparing the two means restricting CORR to this frame's index.
    """
    shared = predictions.index.intersection(meta_model.index)
    predictions = predictions.loc[shared]
    return _by_era(
        era.loc[shared],
        lambda idx: correlation_contribution(
            predictions.loc[idx], meta_model.loc[idx], target.loc[idx].copy()
        ),
    )


def era_max_feature_corr(
    predictions: pd.DataFrame, features: pd.DataFrame, era: pd.Series
) -> pd.DataFrame:
    """Largest absolute feature exposure per era, on the *submitted* artifact.

    Predictions are percentile-ranked within the era first, because that is what
    `rank_normalize` uploads. Equivalent to `numerai_tools.max_feature_correlation`
    column by column, computed as one matrix product instead of a Python loop.
    """

    def largest(idx: pd.Index) -> pd.Series:
        ranked = predictions.loc[idx].rank(pct=True).to_numpy(dtype=float)
        exposures = features.loc[idx].to_numpy(dtype=float)
        corr = _cross_correlation(ranked, exposures)
        return pd.Series(np.nanmax(np.abs(corr), axis=1), index=predictions.columns)

    return _by_era(era, largest)


def era_feature_corr(
    predictions: pd.Series, features: pd.DataFrame, era: pd.Series
) -> pd.DataFrame:
    """Per-era exposure of one prediction column to every feature — era_max_feature_corr's un-reduced input.

    Ranks the prediction within era first, exactly as era_max_feature_corr and
    rank_normalize do, so the correlation this reports is against the shape
    Numerai actually receives. Used to rank features by measured exposure
    (issue #35) rather than to score a submitted artifact.
    """

    def row(idx: pd.Index) -> pd.Series:
        ranked = predictions.loc[idx].rank(pct=True).to_numpy(dtype=float).reshape(-1, 1)
        exposures = features.loc[idx].to_numpy(dtype=float)
        corr = _cross_correlation(ranked, exposures)[0]
        return pd.Series(np.abs(corr), index=features.columns)

    return _by_era(era, row)


def _cross_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of `left` against every column of `right`.

    A column that is constant within the era has no correlation to report, so it
    becomes NaN rather than a divide-by-zero — callers skip it with `nanmax`.
    """
    left = _standardize(left)
    right = _standardize(right)
    return (left.T @ right) / len(left)


def _standardize(values: np.ndarray) -> np.ndarray:
    spread = values.std(0)
    return (values - values.mean(0)) / np.where(spread == 0, np.nan, spread)


def summarize_era_scores(
    era_corr: pd.DataFrame,
    era_mmc: pd.DataFrame,
    era_max_feature_corr: pd.DataFrame,
) -> pd.DataFrame:
    """One row per prediction column — the comparison table.

    `payout` weights CORR and MMC as Numerai does, and takes CORR from the
    meta-model window so both halves are measured over the same eras. `mean_corr`
    over the full validation span is reported alongside, and the two are not
    interchangeable.
    """
    window = era_corr.loc[era_mmc.index]
    summary = pd.DataFrame(
        {
            "eras": len(era_corr),
            "mean_corr": era_corr.mean(),
            "std_corr": era_corr.std(),
            "sharpe": era_corr.mean() / era_corr.std(),
            "smart_sharpe": era_corr.apply(_smart_sharpe),
            "mmc_eras": len(era_mmc),
            "mean_corr_window": window.mean(),
            "mean_mmc": era_mmc.mean(),
            "std_mmc": era_mmc.std(),
            "mmc_sharpe": era_mmc.mean() / era_mmc.std(),
            "max_feature_corr": era_max_feature_corr.mean(),
        }
    )
    summary["payout"] = (
        PAYOUT_CORR_MULTIPLIER * summary["mean_corr_window"]
        + PAYOUT_MMC_MULTIPLIER * summary["mean_mmc"]
    )
    summary.index.name = "config"
    return summary


def _smart_sharpe(corrs: pd.Series) -> float:
    values = corrs.to_numpy(dtype=float)
    return float(values.mean() / (values.std(ddof=1) * _autocorr_penalty(values)))


def era_feature_projection(
    df: pd.DataFrame,
    columns: list[str],
    neutralizers: list[str],
    *,
    era_col: str = "era",
) -> pd.DataFrame:
    """Per-era linear component of every column in `columns`, as `neutralize` computes it.

    Neutralizing at proportion `p` is `scores - p * projection` — affine in `p`,
    and the projection does not depend on `p` at all. So a sweep over proportions
    and blends costs one least-squares solve per era rather than one per
    configuration, which is the difference between minutes and an hour.
    """
    parts = [
        pd.DataFrame(
            project(era_df[columns].to_numpy(dtype=float), era_df[neutralizers].values),
            index=era_df.index,
            columns=columns,
        )
        for _, era_df in df.groupby(era_col, group_keys=False)
    ]
    return pd.concat(parts).loc[df.index]
