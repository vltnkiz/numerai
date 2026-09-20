import pytest

from zemir.config import (
    ENSEMBLE_MODEL_WEIGHTS,
    ENSEMBLE_NEUTRALIZERS,
    PRODUCTION_STRATEGY,
    STRATEGIES,
    XGBOOST_HYPERPARAMS,
    neutralizer_subset_sweep,
    scoring_sweep,
    weight_sweep,
)
from zemir.strategy import BlendSpec, ModelSpec, Neutralization, Strategy

# Frozen expected values — gate 1 of #74's three cheap gates. Captured by
# running this same comparison against the deleted MODEL_NAMES/build_trainers'
# by_name/MODEL_WEIGHTS/NEUTRALIZERS tables and zemir.config.LIVE.xgboost
# before #77 deleted them, confirming STRATEGIES reproduces exactly what those
# tables produced for "linear"/"era_boost"/"ensemble". No fit required: the
# trainers themselves (train_linear, train_xgboost) are untouched code, so
# this is the whole equivalence argument.
#
# #88 moved neutralization onto its own type: `BlendSpec(proportion=0.95)` with
# no neutralizers (meaning "every column the run loaded", which for these
# single-width strategies is `medium`) is now `Neutralization(0.95, "medium")`,
# stating that set instead of leaving it to whatever the caller loaded.


def test_linear_strategy_matches_deleted_tables():
    assert STRATEGIES["linear"] == Strategy(
        models=(ModelSpec(name="linear", features="medium", trainer="ols"),),
        blend=BlendSpec(weights=None, neutralization=Neutralization(0.95, "medium")),
    )


def test_era_boost_strategy_matches_deleted_tables():
    assert STRATEGIES["era_boost"] == Strategy(
        models=(
            ModelSpec(
                name="era_boost",
                features="medium",
                trainer="xgboost",
                params=XGBOOST_HYPERPARAMS,
            ),
        ),
        blend=BlendSpec(weights=None, neutralization=Neutralization(0.95, "medium")),
    )


def test_zemir_01_strategy_matches_deleted_ensemble_table():
    """`zemir_01` is `ensemble` under its new name — #74/#77's one renaming."""
    assert STRATEGIES["zemir_01"] == Strategy(
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
            neutralization=Neutralization(0.95, ENSEMBLE_NEUTRALIZERS),
        ),
    )


def test_production_strategy_is_zemir_01():
    """`PRODUCTION_STRATEGY` is the single line that decides what production ships."""
    assert PRODUCTION_STRATEGY is STRATEGIES["zemir_01"]


def test_no_shipped_strategy_neutralizes_a_model_before_the_blend():
    """#82 ships the capability, not a strategy: per-model neutralization stays unused in `config.py`.

    A strategy sitting in production config that nothing measures is the drift
    `PRODUCTION_BASELINE` was the fossil of. The capability is proven in tests
    that could not be written against the old code (test_blend.py), not shipped.
    """
    assert all(spec.neutralization is None for s in STRATEGIES.values() for spec in s.models)


# --- sweeps: `Strategy` rows a harness cache can be rescored under (#88) -------------------

BASE = STRATEGIES["zemir_01"]


def test_scoring_sweep_names_rows_the_way_the_tables_always_have():
    rows = scoring_sweep(BASE, features="medium")

    assert list(rows) == [
        f"{models}_p{p}"
        for models in ("linear", "era_boost", "linear+era_boost")
        for p in ("0", "0.25", "0.5", "0.75", "1")
    ]


def test_a_zero_proportion_row_has_no_neutralization_at_all():
    rows = scoring_sweep(BASE, features="medium")

    assert rows["linear_p0"].blend.neutralization is None
    assert rows["linear_p0.5"].blend.neutralization == Neutralization(0.5, "medium")


def test_sweep_rows_keep_the_bases_own_models_and_vary_only_the_blend():
    rows = scoring_sweep(BASE, features="medium")

    assert rows["era_boost_p0.75"].models == (BASE.models[1],)
    assert rows["linear+era_boost_p0.5"].models == BASE.models


def test_a_sweep_over_a_model_the_base_lacks_is_refused():
    with pytest.raises(KeyError, match="ghost"):
        scoring_sweep(BASE, features="medium", model_sets=(("ghost",),))


def test_weight_sweep_carries_each_weighting_on_the_blend():
    rows = weight_sweep(BASE, ("linear", "era_boost"), features="medium", resolution=2)

    assert list(rows) == [
        "linear+era_boost_w0-1_p0.95",
        "linear+era_boost_w0.5-0.5_p0.95",
        "linear+era_boost_w1-0_p0.95",
    ]
    assert rows["linear+era_boost_w0.5-0.5_p0.95"].blend == BlendSpec(
        weights={"linear": 0.5, "era_boost": 0.5},
        neutralization=Neutralization(0.95, "medium"),
    )


def test_neutralizer_subset_sweep_compares_the_full_set_with_each_top_k():
    ranked = ["c", "a", "b", "d"]

    rows = neutralizer_subset_sweep(BASE, ranked, (2, 3), features="medium")

    assert list(rows) == [
        "linear+era_boost_p0.95_full",
        "linear+era_boost_p0.95_top2",
        "linear+era_boost_p0.95_top3",
    ]
    assert rows["linear+era_boost_p0.95_full"].blend.neutralization == Neutralization(0.95, "medium")
    assert rows["linear+era_boost_p0.95_top2"].blend.neutralization == Neutralization(0.95, ("c", "a"))
    assert rows["linear+era_boost_p0.95_top3"].blend.weights == dict(ENSEMBLE_MODEL_WEIGHTS)
