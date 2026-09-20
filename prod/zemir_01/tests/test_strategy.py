import json
from dataclasses import asdict

import pytest

from zemir.config import STRATEGIES
from zemir.fitting import fit_strategy
from zemir.pipeline import predict_each
from zemir.strategy import (
    BlendSpec,
    ModelSpec,
    Neutralization,
    Strategy,
    neutralizer_columns,
    required_columns,
    strategy_from_record,
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


def _strategy(
    *feature_sets: str,
    neutralize: tuple[str, ...] | str | None = None,
    model_neutralization: Neutralization | None = None,
) -> Strategy:
    return Strategy(
        models=tuple(
            ModelSpec(name=f"m{i}", features=name, trainer="ols", neutralization=model_neutralization)
            for i, name in enumerate(feature_sets)
        ),
        blend=BlendSpec(
            neutralization=Neutralization(0.5, neutralize) if neutralize is not None else None
        ),
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


def test_required_columns_adds_a_named_neutralizer_set_and_a_per_model_one():
    sets = {"one": ["a", "b"], "wide": ["a", "b", "c", "d"], "other": ["x"]}

    strategy = _strategy(
        "one", neutralize="wide", model_neutralization=Neutralization(1.0, ("y",))
    )

    # Models' union first, then each neutralization's columns in the order the strategy applies them.
    assert required_columns(strategy, sets) == ["a", "b", "y", "c", "d"]


def test_feature_set_names_are_distinct_in_first_seen_order():
    assert _strategy("b", "a", "b").feature_set_names == ("b", "a")


def test_feature_set_names_include_the_sets_a_neutralization_names_but_not_explicit_tuples():
    strategy = _strategy(
        "b", neutralize="wide", model_neutralization=Neutralization(1.0, ("x", "y"))
    )

    assert strategy.feature_set_names == ("b", "wide")


def test_neutralizer_columns_resolve_a_set_name_or_pass_a_tuple_through():
    sets = {"one": ["b", "a"]}

    assert neutralizer_columns(Neutralization(0.5, "one"), sets) == ("b", "a")
    assert neutralizer_columns(Neutralization(0.5, ("z", "y")), sets) == ("z", "y")


def test_an_unresolved_neutralizer_set_names_itself():
    with pytest.raises(KeyError, match="'nope'"):
        neutralizer_columns(Neutralization(0.5, "nope"), {"one": ["a"]})


def test_neutralizations_lists_each_models_then_the_blends():
    strategy = _strategy("a", neutralize="all", model_neutralization=Neutralization(0.2, "sub"))

    assert strategy.neutralizations == (Neutralization(0.2, "sub"), Neutralization(0.5, "all"))
    assert _strategy("a").neutralizations == ()


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_a_recorded_strategy_round_trips_through_fit_config_json(name):
    """`fit_config.json` stores `asdict(strategy)`; the harness rebuilds it to compare fit fields."""
    strategy = STRATEGIES[name]

    assert strategy_from_record(json.loads(json.dumps(asdict(strategy)))) == strategy


def test_a_per_model_neutralization_survives_the_round_trip():
    strategy = _strategy("a", neutralize=("x", "y"), model_neutralization=Neutralization(0.3, "sub"))

    assert strategy_from_record(json.loads(json.dumps(asdict(strategy)))) == strategy


def test_a_record_from_before_neutralization_existed_is_not_a_record():
    old = {
        "models": [{"name": "linear", "features": "medium", "trainer": "ols", "params": {}, "neutralize": None}],
        "blend": {"weights": None, "neutralize": None, "proportion": 0.95},
    }

    with pytest.raises(KeyError):
        strategy_from_record(old)
