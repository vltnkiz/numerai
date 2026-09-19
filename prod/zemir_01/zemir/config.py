"""Named pipeline configurations.

One flat, frozen `PipelineConfig` describes everything about a run except which
models to fit and where to write. Named profiles (`LIVE`, `SMOKE`) are instances
of it; a sweep builds more with `dataclasses.replace(LIVE, ...)`.

Deliberately flat: no sub-configs, and no `run_id` — a config names a
*configuration*, not an instance of running one. The submission slot and the
validation gate live here as plain constants rather than config fields, because
`run_pipeline` neither submits nor gates; only scripts/run_pipeline.py does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from zemir.neutralizer_ranking import MEDIUM_FEATURE_EXPOSURE_RANKING
from zemir.strategy import BlendSpec, ModelSpec, Strategy

# Issue #31: the loop measurably beats a plain single fit (`num_iters=0`) on
# mean_corr/sharpe/smart_sharpe, so it ships as-is. `random_state` pinned only
# to document intent — xgboost already defaulted to seed 0 internally.
#
# `colsample_bytree` is retuned to the feature-set width, not left at a fixed
# fraction (issue #42): the quantity that matters is how many features a tree
# actually samples, and a fraction tuned at `small`'s 42 features means
# something entirely different at `medium`'s 780. Holding the *absolute* count
# roughly constant — `small`'s shipped 0.1 x 42 ~ 4.2 features/tree, so
# 4.2/780 ~ 0.00538, rounded to 0.005 (~3.9 features/tree) — is the heuristic,
# not an independent sweep; #42's width comparison was decisive enough
# (~11.4x `small`'s best payout) not to need one. Retune this whenever
# `feature_set` changes.
XGBOOST_HYPERPARAMS: Mapping[str, object] = {
    "trees_per_step": 50,
    "num_iters": 40,
    "proportion": 0.5,
    "learning_rate": 0.01,
    "max_depth": 5,
    "colsample_bytree": 0.005,
    "random_state": 0,
}

# The dataset's generic `target` alias is not guaranteed to track whichever
# target Numerai currently pays on — v5.3's `target` aliases `target_ender_60`,
# while Numerai has scored payouts against `target_ender_20` specifically since
# 2026-01-01 (confirmed via Numerai's own announcement, not the dataset's own
# metadata). Era-wise Spearman between the two over the 647-era validation
# window is only 0.465 (issue #64) — a real divergence, not noise. See
# CONTEXT.md's `target` alias / payout target distinction.
TARGET_COLUMN = "target_ender_20"

# Read only by scripts/run_pipeline.py — the one entrypoint that submits.
SUBMISSION_MODEL_SLOT = "zemir_01"
# A sanity floor, not a profitability gate (issue #32, per #34): catches a
# clearly broken run (near-zero or negative rank correlation — a bug, not
# weak signal) without tripping on a genuinely bad-but-real week. Set below
# the worst legitimate window measured so far — the harness's recent-octile
# low of 0.0019 (issue #28) — with margin, since that figure is on
# numerai_corr, not this gate's spearman-based mean_corr.
MIN_VALIDATION_MEAN_CORR = 0.001

# Issue #33: round open/close wall-clock timing is not a documented Numerai
# guarantee (their docs disclaim an upper bound on open-time slippage), so
# rather than guess a safe trigger offset, the live run polls Numerai's current
# round before downloading live data (zemir.schedule.round_to_run). Slip topped
# out at ~12 min on the rounds first observed, but round 1359 (2026-09-19)
# opened at 13:04 UTC, 64 min late, after a 45-min budget had given up. 90 min
# covers that with margin and still leaves the task's 150-min limit room for a
# ~22-min fit (a 120-min budget would not). Read only by zemir.schedule, on
# behalf of scripts/run_pipeline.py — the one entrypoint gated on round timing
# (run_experiment.py's harness path always runs, regardless of live round state).
ROUND_OPEN_POLL_INTERVAL_SECONDS = 120
ROUND_OPEN_MAX_WAIT_SECONDS = 90 * 60

# Issue #69: the live fit on Windows peaked at 43.2 GiB resident (full-scale
# `medium` ensemble, sampled every 5 s) — about twice issue #44's 21.48 GiB,
# which measured the linear stage alone. Issue #80 re-measured it: 43.20 GiB on
# the pre-#80 code again (same instrument, same machine), 38.93 GiB once
# `zemir.fitting` owns the fit — the 4.3 GiB gap is the validation window
# copy (3.81 GiB logical) that `run_pipeline` used to build up front and hold
# through the linear stage's float64 upcast, and now builds after the fit.
# The peak is the OLS fit, flat for ~60 s at 38.93 GiB. A mixed-width strategy
# (era_boost on `small`, OLS on `medium`) peaked at 39.29 GiB: +0.36 GiB, the
# extra columns the dataset loads. The machine now doubles as a desktop,
# so scripts/run_pipeline.py refuses to *start* a fit with less than this
# available, leaving ~3 GiB headroom over the measured peak: a clean, retried
# skip beats paging for an hour or a MemoryError fifteen minutes in. Re-measure
# when the feature set, models or training eras change.
MIN_AVAILABLE_MEMORY_GIB = 42


@dataclass(frozen=True)
class PipelineConfig:
    data_version: str = "5.3"
    # v5.3's 780-feature `medium`, not the 42-feature `small` production ran
    # through issue #38. Measured at full scale on the harness's paid metrics:
    # payout 0.016411 vs `small`'s best achievable 0.001436 (~11.4x), with
    # `mean_mmc` flipping from negative to positive across every configuration
    # (#42). `medium` is not a superset of `small` — only 11 of `small`'s 42
    # features appear in it (#41), so this is a different feature basis, not
    # "small plus more". `all` (3,555 features) was ruled out twice on
    # measurement, not on principle: infeasible in memory as first attempted
    # (#42/#45), then fitted successfully after a streaming rewrite and still
    # ~9.7% behind `medium` on payout (#48), as was a linear-at-`all` /
    # era_boost-at-`medium` hybrid (#60).
    feature_set: str = "medium"
    # Overwrites the dataset's `target` alias with this named column right at
    # load (`zemir.data._read_parquet`) — every downstream site still reads
    # plain `"target"` unchanged. Issue #64: switched from the alias's
    # untouched default (`target_ender_60` in v5.3) to the actual payout
    # target, `target_ender_20`.
    target_column: str = TARGET_COLUMN
    max_eras: int | None = None  # None = every era


LIVE = PipelineConfig()

# Exercises the identical code path in seconds instead of an hour. Scores from a
# SMOKE run are meaningless and must never be compared against a LIVE-scale one.
# The `num_iters=2` xgboost speed-up this used to layer on top of `max_eras=40`
# lived on `PipelineConfig.xgboost`, which #77 removed — era_boost's iteration
# count is now on the model's own `ModelSpec.params`, so `smoke()` below is
# what applies it to whichever Strategy a run picks.
SMOKE = replace(LIVE, max_eras=40)


def smoke(strategy: Strategy) -> Strategy:
    """`strategy`, with every model's `num_iters` (if it has one) forced to 2.

    One definition of "smoke" that adapts to whatever specs a strategy holds
    — not a parallel `SMOKE_STRATEGIES` table keyed like the tables #74/#77
    deleted. A model with no `num_iters` (e.g. `ols`) is returned untouched.
    """
    models = tuple(
        replace(m, params={**m.params, "num_iters": 2}) if "num_iters" in m.params else m
        for m in strategy.models
    )
    return replace(strategy, models=models)


# Measured by scripts/sweep_weights.py against runs/harness/20260825T194439Z-ensemble/
# (653 validation eras, neutralization_proportion fixed at 0.95 per issue #29) — issue #30.
# Picked on mean_corr/sharpe/smart_sharpe, not the payout argmax: pure linear
# (w=1.0) tops payout only because it is the fact-1 no-op on rank metrics —
# the degenerate, do-nothing case #28 already flagged, not measured skill.
# 0.3/0.7 is the outright best config in the sweep on corr, sharpe AND
# smart_sharpe at once. A starting default, not a claim: re-sweep whenever a
# model joins, leaves, or the fit changes. `models: [(name, trainer, weight),
# ...]` per #34's structure is expressed here as {model-choice -> {trainer-name
# -> weight}}, one entry per `build_trainers` model choice, rather than a
# rewrite of the trainer selector.
ENSEMBLE_MODEL_WEIGHTS: Mapping[str, float] = {"linear": 0.3, "era_boost": 0.7}

# Re-measured by scripts/sweep_neutralizers.py against
# runs/harness/20260905T205947Z-ensemble/ (780 features, 655 validation eras,
# fit against the corrected target_ender_20 payout target per issue #64),
# ranked by rank_feature_exposure on the shipped linear0.3+era_boost0.7 blend —
# issue #65's re-baseline of #43's K choice.
#
# #43's top390 no longer wins under the corrected target: it now trails both
# the full set (payout −0.003009 for top557 vs −0.003756 full vs −0.004598 for
# top390) and top557, which is the new outright best on payout. The K-sweep
# (93/186/279/390/557/full) stays exactly as noisy and non-monotonic as #35
# originally found — top279 is the worst point in the same sweep top557 wins —
# so this is read as "the top390-shaped answer from #43 didn't survive the
# target correction," not as a newly-found clean elbow. Re-measure K — don't
# carry it over — whenever a model joins, leaves, the fit changes, the
# scoring target changes, or `feature_set` changes.
#
# Decided on payout alone, and that is the whole criterion: feature exposure
# (`max_feature_corr`) is priced nowhere on Numerai Classic — not in payouts,
# burn, the payout factor, stake eligibility, the stake cap, or meta-model
# weighting (issue #66, verified against Numerai's docs; see CONTEXT.md).
# The higher exposure a smaller K accepts is a diagnostic to report, not a
# cost to trade payout against. Do not reintroduce it as a tiebreaker when
# re-measuring K.
_NEUTRALIZER_K = 557
ENSEMBLE_NEUTRALIZERS: tuple[str, ...] = MEDIUM_FEATURE_EXPOSURE_RANKING[:_NEUTRALIZER_K]

# Which models, blended how. Replaces the three tables `MODEL_NAMES` used to
# key — `build_trainers`'s `by_name`, `MODEL_WEIGHTS`, `NEUTRALIZERS` — with
# one entry per named strategy; `ensemble` retires into `zemir_01`, `linear`
# and `era_boost` keep their spellings verbatim (#74/#77). Each spec below
# reads its values off the same measured constants the deleted tables did, so
# the rationale comments stay attached to those constants rather than being
# duplicated here.
STRATEGIES: Mapping[str, Strategy] = {
    "linear": Strategy(
        models=(ModelSpec(name="linear", features="medium", trainer="ols"),),
        blend=BlendSpec(proportion=0.95),
    ),
    "era_boost": Strategy(
        models=(
            ModelSpec(
                name="era_boost",
                features="medium",
                trainer="xgboost",
                params=XGBOOST_HYPERPARAMS,
            ),
        ),
        blend=BlendSpec(proportion=0.95),
    ),
    "zemir_01": Strategy(
        models=(
            ModelSpec(name="linear", features="medium", trainer="ols"),
            ModelSpec(
                name="era_boost",
                features="medium",
                trainer="xgboost",
                params=XGBOOST_HYPERPARAMS,
            ),
        ),
        blend=BlendSpec(
            weights=ENSEMBLE_MODEL_WEIGHTS,
            neutralize=ENSEMBLE_NEUTRALIZERS,
            proportion=0.95,
        ),
    ),
}

# The single line that decides what production ships.
PRODUCTION_STRATEGY = STRATEGIES["zemir_01"]


@dataclass(frozen=True)
class ScoringConfig:
    """One row of the comparison table: what to score, downstream of a fit.

    Separate from `PipelineConfig` because it names a different stage. A
    `PipelineConfig` describes a *fit* — an hour of training, recorded once in
    the harness cache's `fit_config.json` and immutable thereafter. A
    `ScoringConfig` describes what is done to those cached predictions, costs
    seconds, and is swept many times against one fit.

    Flat, like `PipelineConfig`: `proportion = 0.0` *is* "no neutralization"
    (the correction term is exactly zero), so there is no separate on/off field.
    """

    name: str
    models: tuple[str, ...]
    neutralization_proportion: float = 0.0
    # Neutralize each model's raw prediction before the per-era rank-blend, rather
    # than neutralizing the blended result. Linear regression's prediction lies
    # exactly in the neutralizers' span (issue #24, established fact 1), so
    # neutralizing it is a positive rescaling that `combine_predictions`'s rank
    # transform is invariant to — this only changes non-linear models in `models`.
    neutralize_before_blend: bool = False
    # Parallel to `models`; None = equal weight (combine_predictions' default).
    weights: tuple[float, ...] | None = None
    # None = every training feature (today's default, per pipeline.py). A
    # specific subset to project onto instead — issue #35.
    neutralizers: tuple[str, ...] | None = None


def scoring_sweep(
    model_sets: tuple[tuple[str, ...], ...] = (
        ("linear",),
        ("era_boost",),
        ("linear", "era_boost"),
    ),
    proportions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[ScoringConfig]:
    """Every combination of models and neutralization proportion, named `<models>_p<prop>`."""
    return [
        ScoringConfig(f"{'+'.join(models)}_p{proportion:g}", models, proportion)
        for models in model_sets
        for proportion in proportions
    ]


def pre_blend_scoring_sweep(
    proportions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> list[ScoringConfig]:
    """Neutralize era_boost before blending with (untouched) linear, at each proportion.

    Answers "neutralize XGBoost only, then blend" from issue #29: linear is
    included in `models` for a correct rank-blend denominator, but pre-blend
    neutralization is a no-op on it (see `ScoringConfig.neutralize_before_blend`),
    so this measures the XGBoost-only-neutralized blend, not a three-way split.
    """
    return [
        ScoringConfig(
            f"linear+era_boost_pre_p{proportion:g}",
            ("linear", "era_boost"),
            proportion,
            neutralize_before_blend=True,
        )
        for proportion in proportions
    ]


def _simplex_grid(n: int, resolution: int) -> list[tuple[float, ...]]:
    """Every way to divide `resolution` units among `n` non-negative weights.

    Generalizes past two models: at `n=2` this is the familiar `resolution + 1`
    points along an edge; at `n=4, resolution=10` it's 286 points — still cheap,
    since `score_configs` only recombines cached predictions, never refits.
    """
    if n == 1:
        return [(1.0,)]

    def _parts(remaining: int, slots: int):
        if slots == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in _parts(remaining - i, slots - 1):
                yield (i, *rest)

    return [tuple(p / resolution for p in combo) for combo in _parts(resolution, n)]


def weight_sweep(
    models: tuple[str, ...],
    *,
    resolution: int = 10,
    neutralization_proportion: float = 0.95,
) -> list[ScoringConfig]:
    """Every weighting of `models` on a resolution-`R` simplex grid, one proportion fixed.

    The reusable primitive behind "how much of each model belongs in the
    blend" (issue #30) — sweeps weight, not model set or proportion, so it
    composes with `scoring_sweep`'s job rather than duplicating it. Reused
    as-is whenever a model joins, leaves, or the fit changes.
    """
    return [
        ScoringConfig(
            f"{'+'.join(models)}_w{'-'.join(f'{w:g}' for w in weights)}_p{neutralization_proportion:g}",
            models,
            neutralization_proportion,
            weights=weights,
        )
        for weights in _simplex_grid(len(models), resolution)
    ]


def neutralizer_subset_sweep(
    ranked_features: list[str],
    ks: tuple[int, ...],
    *,
    models: tuple[str, ...] = ("linear", "era_boost"),
    weights: tuple[float, ...] | None = (0.3, 0.7),
    neutralization_proportion: float = 0.95,
) -> list[ScoringConfig]:
    """Compare full-feature neutralization against the top-K most-exposed features, by K.

    `ranked_features` must already be sorted most- to least-exposed (see
    `zemir.harness.rank_feature_exposure`) — this only slices it. The full set
    (`neutralizers=None`, today's status quo) is always included as the
    baseline row (issue #35).
    """
    full = ScoringConfig(
        f"{'+'.join(models)}_p{neutralization_proportion:g}_full",
        models,
        neutralization_proportion,
        weights=weights,
    )
    subsets = [
        ScoringConfig(
            f"{'+'.join(models)}_p{neutralization_proportion:g}_top{k}",
            models,
            neutralization_proportion,
            weights=weights,
            neutralizers=tuple(ranked_features[:k]),
        )
        for k in ks
    ]
    return [full, *subsets]


# Today's live submission — linear only, neutralized at 0.5. The row every later
# comparison on this map is measured against; a member of `scoring_sweep()`, not
# a duplicate of one.
PRODUCTION_BASELINE = "linear_p0.5"
