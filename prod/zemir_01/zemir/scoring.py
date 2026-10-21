from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr
from scipy.stats import spearmanr


def era_spearman(
    df: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str = "prediction",
    era_col: str = "era",
) -> pd.Series:
    """Plain per-era Spearman correlation — *not* Numerai's paid CORR.

    The paid measure is `era_numerai_corr`, and the two are not
    interchangeable: on the same production run this reports 0.0164 on the
    raw blend against a paid 0.0116 on the submitted blend, so the gap is one
    of metric *and* of artifact.

    Kept for exactly one reason: `score_log.jsonl`'s schema 1 entries recorded
    `mean_corr`, `sharpe` and `smart_sharpe` in this vocabulary since issue
    #68, and the raw blend's Spearman series continues through every later
    schema so they stay comparable — against `LEGACY_SPEARMAN_TARGET`, the
    target those entries were taken on, not the scoring target (#105). Once
    the 365-day prune has removed the last schema 1 entry, delete this,
    `summarize_era_spearman` and `LEGACY_SPEARMAN_TARGET` (#98). Nothing new
    should be measured on it (issue #96).

    The series continues only approximately: validation is refreshed weekly
    (#104), so the frame grows past the 655 eras every schema 1 entry was
    measured on. Fresh data won over continuity. Where an exact comparison
    matters, recompute both sides on the current copy.
    """
    corrs = df.groupby(era_col).apply(
        lambda d: spearmanr(d[prediction_col], d[target_col])[0]
    )
    return corrs.loc[sorted(corrs.index, key=int)]


def _autocorr_penalty(x: np.ndarray) -> float:
    n = len(x)
    p = np.corrcoef(x[:-1], x[1:])[0, 1]
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p**i for i in range(1, n)]))


@dataclass(frozen=True)
class EraSpearmanScore:
    """The three legacy `score_log` columns, and the per-era series behind them.

    Named for the correlation it holds, so it cannot be mistaken for a row of
    `summarize_era_scores` — which reports `mean_corr`, `sharpe` and
    `smart_sharpe` under the very same spellings, computed on the paid CORR.
    Replaces `ValidationScore`, whose name said which *dataset* it covered and
    never which *correlation* it held (issue #96).
    """

    era_spearman: pd.Series
    mean_corr: float
    std_corr: float
    sharpe: float
    smart_sharpe: float


def summarize_era_spearman(corrs: pd.Series) -> EraSpearmanScore:
    """Summarize a per-era Spearman series, arithmetic for arithmetic as `score_validation` did.

    Shares `_smart_sharpe` with `summarize_era_scores`, which is what ends
    `_autocorr_penalty` being computed down two separate paths. Verified
    bit-identical to the retired inline form on both the synthetic fixture and
    the real 655-era production series (issue #96).
    """
    return EraSpearmanScore(
        era_spearman=corrs,
        mean_corr=float(corrs.mean()),
        std_corr=float(corrs.std()),
        sharpe=float(corrs.mean() / corrs.std()),
        smart_sharpe=_smart_sharpe(corrs),
    )


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
# `era_spearman` above uses plain Spearman, which is not what Numerai pays
# on. `validation_payout_proxy_60` prices CORR and MMC with `PAYOUT_MULTIPLIERS`,
# the formula Numerai pays rounds from 1343 on — MMC carries five times the
# weight of CORR — so a comparison made on Spearman alone can pick the wrong
# architecture. It is a ranking proxy for the harness and never a prediction
# of what a round pays: that comes from Numerai (`zemir.live_scores`). These
# wrap `numerai-tools`, Numerai's own reference implementation, rather than
# reimplementing the formulas.
#
# Everything below this line is the vocabulary the live run and the harness
# share, composed by `score_predictions`. Everything above it survives only to
# keep `score_log.jsonl`'s existing entries readable — see issue #96.
#
# All three are invariant to any strictly monotone transform of the
# predictions, `rank_normalize` included: they rank internally. Verified
# exactly (delta 0.00e+00) — see issue #26.
# ---------------------------------------------------------------------------

# The payout target: what Numerai scores CORR and MMC against, from the
# Ender-60 cutover at round 1343 (#105). Every paid metric in this repo — the
# gate, the history log, the harness — is measured against it, whatever
# target a model was fitted on (`ModelSpec.target`). Numerai changes it on its
# own schedule, so confirm it against Numerai's scoring docs, never against
# the dataset's generic `target` alias (#64).
SCORING_TARGET = "target_ender_60"
# The target `era_spearman`'s series has always been measured against, kept
# only so the history log's legacy Spearman series runs unbroken.
LEGACY_SPEARMAN_TARGET = "target_ender_20"
# Numerai's payout multipliers from round 1343 on. Numerai lists them per
# round, so `zemir.live_scores` prices each round with that round's own; this
# is the current formula, which the harness proxy and every round's
# `reference_payout_score` share. Update it when Numerai lists new
# multipliers on new rounds.
PAYOUT_MULTIPLIERS = {"corr60": 3.0, "mmc60": 15.0}
# The harness's ranking column. Named for what it is, so no table can pass it
# off as Numerai's payout, and with its horizon, so no table can pass it off as
# the retired 20-day `validation_payout_proxy` (#105, #106).
VALIDATION_PAYOUT_PROXY = "validation_payout_proxy_60"
# Consecutive eras whose 60-day targets overlap: an era is a week, and 60
# business days are twelve of them. The bootstrap resamples blocks this long so
# its interval does not treat overlapping eras as independent draws (#105).
BOOTSTRAP_BLOCK_ERAS = 12


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
    corr_by_era: pd.DataFrame,
    mmc_by_era: pd.DataFrame,
    exposure_by_era: pd.DataFrame,
) -> pd.DataFrame:
    """One row per prediction column — the comparison table.

    `VALIDATION_PAYOUT_PROXY` weights CORR and MMC with `PAYOUT_MULTIPLIERS`,
    and takes CORR from the meta-model window so both halves are measured over the same eras. `mean_corr`
    over the full validation span is reported alongside, and the two are not
    interchangeable.

    `mean_corr`, `sharpe` and `smart_sharpe` here are computed on Numerai's
    **paid** CORR. `score_log.jsonl` records three columns of those same names
    taken from `summarize_era_spearman`; they are different quantities, and
    live under separate keys for exactly that reason (issue #96).

    Parameters are named for the shape they carry rather than the functions
    that produce them, which is what stops them shadowing `era_numerai_corr`,
    `era_mmc` and `era_max_feature_corr` inside this body.
    """
    window = corr_by_era.loc[mmc_by_era.index]
    summary = pd.DataFrame(
        {
            "eras": len(corr_by_era),
            "mean_corr": corr_by_era.mean(),
            "std_corr": corr_by_era.std(),
            "sharpe": corr_by_era.mean() / corr_by_era.std(),
            "smart_sharpe": corr_by_era.apply(_smart_sharpe),
            "mmc_eras": len(mmc_by_era),
            "mean_corr_window": window.mean(),
            "mean_mmc": mmc_by_era.mean(),
            "std_mmc": mmc_by_era.std(),
            "mmc_sharpe": mmc_by_era.mean() / mmc_by_era.std(),
            "max_feature_corr": exposure_by_era.mean(),
        }
    )
    summary[VALIDATION_PAYOUT_PROXY] = _proxy(summary["mean_corr_window"], summary["mean_mmc"])
    summary.index.name = "config"
    return summary


def _proxy(corr: pd.Series | float, mmc: pd.Series | float) -> pd.Series | float:
    return PAYOUT_MULTIPLIERS["corr60"] * corr + PAYOUT_MULTIPLIERS["mmc60"] * mmc


def proxy_lead_interval(
    corr_by_era: pd.DataFrame,
    mmc_by_era: pd.DataFrame,
    baseline: str,
    *,
    level: float = 0.95,
    block: int = BOOTSTRAP_BLOCK_ERAS,
    draws: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each column's `VALIDATION_PAYOUT_PROXY` lead over `baseline`, with a moving-block bootstrap interval.

    Measured over the meta-model window, as the proxy is. Resamples runs of
    `block` consecutive eras, so eras whose targets overlap are drawn together
    rather than counted as independent: at 60 days, ~89 window eras hold only
    about eight independent samples, and an interval that ignored that would
    be far too narrow (#105). Reported, never applied: whether a lead is worth
    shipping is decided in the PR that would ship it.

    Columns `lead`, `lead_low`, `lead_high`; `baseline`'s own row is all zero.
    """
    window = corr_by_era.loc[mmc_by_era.index]
    lead = _proxy(window, mmc_by_era).sub(_proxy(window, mmc_by_era)[baseline], axis=0)
    values = lead.to_numpy(dtype=float)
    n = len(values)
    block = min(block, n)
    starts_per_draw = -(-n // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(draws, starts_per_draw))
    rows = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    means = values[rows].mean(axis=1)
    tail = (1 - level) / 2
    return pd.DataFrame(
        {
            "lead": values.mean(axis=0),
            "lead_low": np.quantile(means, tail, axis=0),
            "lead_high": np.quantile(means, 1 - tail, axis=0),
        },
        index=lead.columns,
    ).rename_axis("config")


def _smart_sharpe(corrs: pd.Series) -> float:
    values = corrs.to_numpy(dtype=float)
    return float(values.mean() / (values.std(ddof=1) * _autocorr_penalty(values)))


@dataclass(frozen=True)
class PredictionScores:
    """Everything one set of prediction columns scores, in the paid vocabulary.

    The four artifacts `score_configs` has always produced, minus the `run_dir`
    it used to echo back: the caller already holds that, and dropping it is
    what lets the live run receive this same type without inventing a
    directory it has no use for.
    """

    summary: pd.DataFrame
    corr_by_era: pd.DataFrame
    mmc_by_era: pd.DataFrame
    exposure_by_era: pd.DataFrame


def score_predictions(
    predictions: pd.DataFrame,
    target: pd.Series,
    era: pd.Series,
    *,
    meta_model: pd.Series,
    features: pd.DataFrame,
) -> PredictionScores:
    """Score prediction columns on Numerai's paid metrics — the one composition both paths call.

    The live run and the harness measure the same quantities through this
    function, which is what makes a live run's numbers comparable to a harness
    table at all (issue #93). `score_configs` hands it columns blended from a
    cache; the live run hands it its own validation blend. Neither concept
    appears here: this takes frames and returns numbers.

    Pure by design — it writes nothing. Artifact layout stays the caller's,
    so the harness keeps its `scores.csv`/`era_*.csv` set and the live run is
    free to record something else without this function growing a second shape.

    `predictions`, `target`, `era` and `features` share one index.
    `rank_normalize` is deliberately not applied: all three measures rank
    internally, so it is provably free (issue #26).

    The live path needs `numerai_corr` *before* it submits and the rest only
    after, so it calls `era_numerai_corr` directly for the gate and this
    afterwards for the record — recomputing CORR for 22 s on the far side of
    the submission rather than handing this function a half-filled table.
    """
    corr_by_era = era_numerai_corr(predictions, target, era)
    mmc_by_era = era_mmc(predictions, target, era, meta_model)
    exposure_by_era = era_max_feature_corr(predictions, features, era)
    return PredictionScores(
        summary=summarize_era_scores(corr_by_era, mmc_by_era, exposure_by_era),
        corr_by_era=corr_by_era,
        mmc_by_era=mmc_by_era,
        exposure_by_era=exposure_by_era,
    )
