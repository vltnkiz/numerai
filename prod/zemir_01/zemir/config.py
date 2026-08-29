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
from dataclasses import dataclass, field, replace

from zemir.models import Trainer, train_linear, train_xgboost

# Issue #31: the loop measurably beats a plain single fit (`num_iters=0`) on
# mean_corr/sharpe/smart_sharpe, so it ships as-is. `random_state` pinned only
# to document intent — xgboost already defaulted to seed 0 internally.
XGBOOST_HYPERPARAMS: Mapping[str, object] = {
    "trees_per_step": 50,
    "num_iters": 40,
    "proportion": 0.5,
    "learning_rate": 0.01,
    "max_depth": 5,
    "colsample_bytree": 0.1,
    "random_state": 0,
}

# Read only by scripts/run_pipeline.py — the one entrypoint that submits.
SUBMISSION_MODEL_SLOT = "zemir_01"
# A sanity floor, not a profitability gate (issue #32, per #34): catches a
# clearly broken run (near-zero or negative rank correlation — a bug, not
# weak signal) without tripping on a genuinely bad-but-real week. Set below
# the worst legitimate window measured so far — the harness's recent-octile
# low of 0.0019 (issue #28) — with margin, since that figure is on
# numerai_corr, not this gate's spearman-based mean_corr.
MIN_VALIDATION_MEAN_CORR = 0.001

MODEL_NAMES = ["linear", "era_boost", "ensemble"]


@dataclass(frozen=True)
class PipelineConfig:
    data_version: str = "5.3"
    feature_set: str = "small"
    # Full-blend neutralization, not the linear-only no-op 0.5 used to be:
    # measured optimum is p=1.0 on the harness sweep (issue #29), backed off to
    # 0.95 for margin against validate_predictions' zero-variance guard.
    neutralization_proportion: float = 0.95
    max_eras: int | None = None  # None = every era
    xgboost: Mapping[str, object] = field(
        default_factory=lambda: dict(XGBOOST_HYPERPARAMS)
    )
    # None = equal weight (combine_predictions' default). Set per `--model` by
    # MODEL_WEIGHTS below; a lone model ignores this entirely.
    model_weights: Mapping[str, float] | None = None
    # None = every training feature (pipeline.py's own no-argument default). Set
    # per `--model` by NEUTRALIZERS below.
    neutralizers: tuple[str, ...] | None = None


LIVE = PipelineConfig()

# Exercises the identical code path in seconds instead of an hour. Scores from a
# SMOKE run are meaningless and must never be compared against a LIVE-scale one.
SMOKE = replace(
    LIVE,
    max_eras=40,
    xgboost={**XGBOOST_HYPERPARAMS, "num_iters": 2},
)


def build_trainers(model: str, config: PipelineConfig) -> dict[str, Trainer]:
    """Trainers for one model choice, using `config`'s hyperparameters."""

    def linear(X, y, era):
        return train_linear(X, y)

    def era_boost(X, y, era):
        return train_xgboost(X, y, era, **config.xgboost)

    by_name: dict[str, dict[str, Trainer]] = {
        "linear": {"linear": linear},
        "era_boost": {"era_boost": era_boost},
        "ensemble": {"linear": linear, "era_boost": era_boost},
    }
    if model not in by_name:
        raise ValueError(f"unknown model {model!r} (have: {MODEL_NAMES})")
    return by_name[model]


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

MODEL_WEIGHTS: Mapping[str, Mapping[str, float] | None] = {
    "linear": None,
    "era_boost": None,
    "ensemble": ENSEMBLE_MODEL_WEIGHTS,
}

# Measured by scripts/sweep_neutralizers.py against the same
# runs/harness/20260825T194439Z-ensemble cache as ENSEMBLE_MODEL_WEIGHTS, ranked
# by rank_feature_exposure on the shipped linear0.3+era_boost0.7 blend — issue
# #35. The top-10-most-exposed subset was the best-payout row in a K-sweep
# (5/10/15/21/30/full) that was otherwise noisy and non-monotonic in K: every
# subset beat the full 42-feature set on payout/mean_corr/mmc, at the cost of
# 5-8x higher max_feature_corr on the features it stops protecting against.
# Chosen as the empirical winner despite that noise — not because the curve
# points at K=10 specifically. Re-measure whenever a model joins, leaves, or
# the fit changes, same as ENSEMBLE_MODEL_WEIGHTS.
ENSEMBLE_NEUTRALIZERS: tuple[str, ...] = (
    "feature_petty_upraised_caddice",
    "feature_jewish_stained_disembowelment",
    "feature_snakiest_somalian_wavelet",
    "feature_willful_sere_chronobiology",
    "feature_departmental_inimitable_sentencer",
    "feature_pottier_unmanly_collyrium",
    "feature_antistrophic_striate_conscriptionist",
    "feature_unbreakable_constraining_hegelianism",
    "feature_transisthmian_yogic_linden",
    "feature_stretchy_spiniest_fizgig",
)

NEUTRALIZERS: Mapping[str, tuple[str, ...] | None] = {
    "linear": None,
    "era_boost": None,
    "ensemble": ENSEMBLE_NEUTRALIZERS,
}


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
