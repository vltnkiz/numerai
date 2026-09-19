from zemir.config import (
    ENSEMBLE_MODEL_WEIGHTS,
    ENSEMBLE_NEUTRALIZERS,
    PRODUCTION_STRATEGY,
    STRATEGIES,
    XGBOOST_HYPERPARAMS,
)
from zemir.strategy import BlendSpec, ModelSpec, Strategy

# Frozen expected values — gate 1 of #74's three cheap gates. Captured by
# running this same comparison against the deleted MODEL_NAMES/build_trainers'
# by_name/MODEL_WEIGHTS/NEUTRALIZERS tables and zemir.config.LIVE.xgboost
# before #77 deleted them, confirming STRATEGIES reproduces exactly what those
# tables produced for "linear"/"era_boost"/"ensemble". No fit required: the
# trainers themselves (train_linear, train_xgboost) are untouched code, so
# this is the whole equivalence argument.


def test_linear_strategy_matches_deleted_tables():
    assert STRATEGIES["linear"] == Strategy(
        models=(ModelSpec(name="linear", features="medium", trainer="ols"),),
        blend=BlendSpec(weights=None, neutralize=None, proportion=0.95),
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
        blend=BlendSpec(weights=None, neutralize=None, proportion=0.95),
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
            neutralize=ENSEMBLE_NEUTRALIZERS,
            proportion=0.95,
        ),
    )


def test_production_strategy_is_zemir_01():
    """`PRODUCTION_STRATEGY` is the single line that decides what production ships."""
    assert PRODUCTION_STRATEGY is STRATEGIES["zemir_01"]
