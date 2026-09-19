from zemir.fitting import fit_strategy
from zemir.pipeline import predict_each
from zemir.strategy import (
    BlendSpec,
    ModelSpec,
    Strategy,
    required_columns,
    union_columns,
)

from factories import FEATURE_SETS, make_dataset


def test_two_era_boosts_at_different_depths_fit_and_predict_as_distinct_models():
    """The expressiveness test #76 exists for.

    Today's `build_trainers` returns a `dict[str, Trainer]` keyed by model
    name, so two `era_boost` specs collide on the key `"era_boost"` and this
    cannot even be expressed. Keying by each `ModelSpec`'s own `name` instead
    is what makes #74's headline claim a red-green fact.
    """
    dataset = make_dataset()
    strategy = Strategy(
        models=(
            ModelSpec(
                name="era_boost_shallow",
                features="medium",
                trainer="xgboost",
                params={
                    "trees_per_step": 5,
                    "num_iters": 0,
                    "max_depth": 1,
                    "colsample_bytree": 1.0,
                },
            ),
            ModelSpec(
                name="era_boost_deep",
                features="medium",
                trainer="xgboost",
                params={
                    "trees_per_step": 5,
                    "num_iters": 0,
                    "max_depth": 4,
                    "colsample_bytree": 1.0,
                },
            ),
        ),
        blend=BlendSpec(),
    )

    fit = fit_strategy(strategy, lambda: dataset.train, feature_sets=FEATURE_SETS)
    predictions = predict_each(fit.models, dataset.validation)

    assert set(predictions) == {"era_boost_shallow", "era_boost_deep"}
    assert not predictions["era_boost_shallow"].equals(predictions["era_boost_deep"])


def _strategy(*feature_sets: str, neutralize: tuple[str, ...] | None = None) -> Strategy:
    return Strategy(
        models=tuple(
            ModelSpec(name=f"m{i}", features=name, trainer="ols")
            for i, name in enumerate(feature_sets)
        ),
        blend=BlendSpec(neutralize=neutralize),
    )


def test_union_columns_is_each_column_once_in_first_seen_order():
    sets = {"one": ["b", "a"], "two": ["a", "c"]}

    assert union_columns(_strategy("one", "two"), sets) == ("b", "a", "c")


def test_union_of_models_naming_one_set_is_that_sets_own_order():
    # Tree stages sample columns by position: the shipped strategies (every model
    # on `medium`) must see `medium`'s columns exactly as `features.json` lists them.
    sets = {"medium": ["z", "y", "x"]}

    assert union_columns(_strategy("medium", "medium"), sets) == ("z", "y", "x")


def test_required_columns_adds_blend_neutralizers_outside_the_union():
    sets = {"one": ["a", "b"]}

    strategy = _strategy("one", neutralize=("b", "n1", "n2"))

    assert required_columns(strategy, sets) == ["a", "b", "n1", "n2"]


def test_feature_set_names_are_distinct_in_first_seen_order():
    assert _strategy("b", "a", "b").feature_set_names == ("b", "a")
