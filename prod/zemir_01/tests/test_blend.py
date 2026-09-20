import numpy as np
import pandas as pd
import pytest

import zemir.blend as blend_module
from zemir.blend import blend_strategies, combine_predictions, project
from zemir.strategy import BlendSpec, ModelSpec, Neutralization, Strategy

FEATURES = ["f1", "f2", "f3", "f4"]
FEATURE_SETS = {"pair": ("f1", "f2"), "all4": tuple(FEATURES)}


def _spec(name: str, neutralization: Neutralization | None = None) -> ModelSpec:
    return ModelSpec(name=name, features="all4", trainer="ols", neutralization=neutralization)


def _strategy(*specs: ModelSpec, weights=None, neutralization=None) -> Strategy:
    return Strategy(models=specs, blend=BlendSpec(weights=weights, neutralization=neutralization))


def _frames(n_eras: int = 3, rows_per_era: int = 40, seed: int = 0):
    """Two models' raw predictions, `f1..f4` int8 features (as Numerai ships them), and an era column."""
    rng = np.random.default_rng(seed)
    n = n_eras * rows_per_era
    index = pd.Index([f"n{i:04d}" for i in range(n)], name="id")
    era = pd.Series([f"{1 + i // rows_per_era:04d}" for i in range(n)], index=index, name="era")
    features = pd.DataFrame(
        {c: rng.integers(0, 5, size=n).astype("int8") for c in FEATURES}, index=index
    )
    # `a` leans on f1, `b` on f2 and f3, plus noise, so each has something for a neutralization to remove.
    predictions = pd.DataFrame(
        {
            "a": features["f1"] + rng.normal(size=n),
            "b": features["f2"] - 2.0 * features["f3"] + rng.normal(size=n),
        },
        index=index,
    )
    return predictions, features, era


def _neutralized(values: pd.Series, features: pd.DataFrame, era: pd.Series, columns, proportion: float):
    """`values` with `proportion` of its per-era linear component on `columns` removed — the definition, spelled out."""
    out = pd.Series(index=values.index, dtype=float)
    for label in era.unique():
        rows = era == label
        scores = values[rows].to_numpy(dtype=float).reshape(-1, 1)
        exposures = features.loc[rows, list(columns)].values
        out[rows] = (scores - proportion * project(scores, exposures)).ravel()
    return out


# --- combine_predictions -----------------------------------------------------------------


def test_combine_predictions_returns_a_lone_model_untouched():
    preds = pd.Series([0.1, 0.9, 0.3])
    era = pd.Series(["0001", "0001", "0001"])

    combined = combine_predictions({"linear": preds}, era)

    pd.testing.assert_series_equal(combined, preds.rename("prediction"))


def test_combine_predictions_blends_per_era_ranks_by_weight():
    # One era, two models. A=[10,20,30,40] ranks to [.25,.5,.75,1.0]; B is A
    # reversed and so ranks to [1.0,.75,.5,.25]. At weights 0.75/0.25 the
    # blend is 0.75*A_rank + 0.25*B_rank.
    era = pd.Series(["0001"] * 4)
    a = pd.Series([10.0, 20.0, 30.0, 40.0])
    b = pd.Series([40.0, 30.0, 20.0, 10.0])

    combined = combine_predictions({"a": a, "b": b}, era, {"a": 0.75, "b": 0.25})

    assert combined.tolist() == pytest.approx([0.4375, 0.5625, 0.6875, 0.8125])


def test_combine_predictions_ranks_within_each_era_independently():
    # Same relative pattern in both eras, but at very different magnitudes —
    # a rank blend should produce the identical result in each, since it
    # never compares magnitudes across eras.
    era = pd.Series(["0001", "0001", "0002", "0002"])
    a = pd.Series([1.0, 2.0, 100.0, 200.0])
    b = pd.Series([2.0, 1.0, 200.0, 100.0])

    combined = combine_predictions({"a": a, "b": b}, era)

    assert combined.tolist() == pytest.approx([0.75, 0.75, 0.75, 0.75])


# --- one strategy: what the live run asks for ---------------------------------------------


def _one_model(values, era, features_by_column, neutralization):
    """A lone model's prediction run through `blend_strategies` with `neutralization` on the blend."""
    index = pd.RangeIndex(len(values))
    predictions = pd.DataFrame({"m": values}, index=index)
    features = pd.DataFrame(features_by_column, index=index)
    strategy = _strategy(_spec("m"), neutralization=neutralization)
    return blend_strategies(
        {"s": strategy}, predictions, features, pd.Series(era, index=index), {}
    )["s"]


def test_a_blend_with_no_neutralization_is_the_plain_blend():
    predictions, features, era = _frames()
    strategy = _strategy(_spec("a"), _spec("b"), weights={"a": 0.3, "b": 0.7})

    result = blend_strategies({"s": strategy}, predictions, features, era, FEATURE_SETS)["s"]

    expected = combine_predictions(dict(predictions.items()), era, {"a": 0.3, "b": 0.7})
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_zero_proportion_neutralization_changes_nothing():
    values = [0.1, 0.4, 0.6, 0.9]

    neutralized = _one_model(values, ["0001"] * 4, {"f1": [1.0, 2.0, 3.0, 4.0]}, Neutralization(0.0, ("f1",)))

    assert neutralized.tolist() == values


def test_full_proportion_removes_a_perfectly_linear_exposure():
    # The prediction is exactly `f1`, so a full-proportion projection onto `f1`
    # explains all of it — the residual should be ~0 everywhere.
    line = [1.0, 2.0, 3.0, 4.0]

    neutralized = _one_model(line, ["0001"] * 4, {"f1": line}, Neutralization(1.0, ("f1",)))

    assert neutralized.abs().max() < 1e-8


def test_each_era_is_neutralized_independently():
    # Era "0001" is perfectly explained by f1 (residual ~0 at full proportion).
    # Era "0002" has a *constant* f1, so nothing there is explained by it —
    # each row's residual is just its own deviation from the era's mean
    # prediction (6.0), regardless of era "0001"'s own f1 relationship.
    era = ["0001", "0001", "0002", "0002"]

    neutralized = _one_model(
        [1.0, 2.0, 5.0, 7.0], era, {"f1": [1.0, 2.0, 3.0, 3.0]}, Neutralization(1.0, ("f1",))
    )

    assert neutralized.iloc[:2].abs().max() < 1e-8
    assert neutralized.iloc[2] == pytest.approx(-1.0)
    assert neutralized.iloc[3] == pytest.approx(1.0)


def test_the_live_path_is_the_definition_of_neutralizing_exactly():
    """Gate 1 of #88: the one-strategy call equals `scores - p * project(scores, exposures)`, bit for bit.

    That expression, per era, on a single column, is what `neutralize_predictions`
    computed before this module existed — pinned here so a refactor of the grouped
    solve cannot move a live prediction by a last bit.
    """
    predictions, features, era = _frames()
    strategy = _strategy(_spec("a"), _spec("b"), neutralization=Neutralization(0.95, ("f1", "f2", "f3")))

    result = blend_strategies({"live": strategy}, predictions, features, era, FEATURE_SETS)["live"]

    blended = combine_predictions(dict(predictions.items()), era)
    expected = _neutralized(blended, features, era, ("f1", "f2", "f3"), 0.95)
    pd.testing.assert_series_equal(result, expected, check_names=False, check_exact=True)


# --- per-model neutralization: what the old code could not express -----------------------


def test_neutralizing_one_model_before_the_blend_is_the_old_pre_blend_sweep():
    """`ScoringConfig.neutralize_before_blend` is deleted; this is what it computed, on the spec that replaces it.

    Its own docstring described the measurement as "neutralize XGBoost only,
    then blend": neutralizing a linear model's prediction is a rescaling the
    rank blend ignores, so only the non-linear model mattered. Here `b` stands
    for that model and is the only one given a neutralization.
    """
    predictions, features, era = _frames()
    strategy = _strategy(
        _spec("a"),
        _spec("b", Neutralization(0.5, ("f1", "f2", "f3", "f4"))),
        weights={"a": 0.3, "b": 0.7},
    )

    result = blend_strategies({"s": strategy}, predictions, features, era, FEATURE_SETS)["s"]

    b_neutralized = _neutralized(predictions["b"], features, era, FEATURES, 0.5)
    expected = combine_predictions({"a": predictions["a"], "b": b_neutralized}, era, {"a": 0.3, "b": 0.7})
    pd.testing.assert_series_equal(result, expected, check_names=False, check_exact=True)


def test_each_model_and_the_blend_can_neutralize_differently():
    """Three stages, three strengths, three feature sets — none of which `ScoringConfig` could say.

    `a` at 0.9 against `f1` alone, `b` at 0.4 against `f2`+`f3`, then the blend at
    0.7 against everything. The old configuration had one proportion and one
    neutralizer set for the whole strategy, applied uniformly.
    """
    predictions, features, era = _frames()
    strategy = _strategy(
        _spec("a", Neutralization(0.9, ("f1",))),
        _spec("b", Neutralization(0.4, ("f2", "f3"))),
        weights={"a": 0.5, "b": 0.5},
        neutralization=Neutralization(0.7, tuple(FEATURES)),
    )

    result = blend_strategies({"s": strategy}, predictions, features, era, FEATURE_SETS)["s"]

    per_model = {
        "a": _neutralized(predictions["a"], features, era, ("f1",), 0.9),
        "b": _neutralized(predictions["b"], features, era, ("f2", "f3"), 0.4),
    }
    blended = combine_predictions(per_model, era, {"a": 0.5, "b": 0.5})
    expected = _neutralized(blended, features, era, FEATURES, 0.7)
    pd.testing.assert_series_equal(result, expected, check_names=False, check_exact=True)

    # ...and each stage is doing something: dropping any one of them changes the answer.
    for dropped in (
        _strategy(
            _spec("a"), _spec("b", Neutralization(0.4, ("f2", "f3"))),
            weights={"a": 0.5, "b": 0.5}, neutralization=Neutralization(0.7, tuple(FEATURES)),
        ),
        _strategy(
            _spec("a", Neutralization(0.9, ("f1",))), _spec("b"),
            weights={"a": 0.5, "b": 0.5}, neutralization=Neutralization(0.7, tuple(FEATURES)),
        ),
        _strategy(
            _spec("a", Neutralization(0.9, ("f1",))), _spec("b", Neutralization(0.4, ("f2", "f3"))),
            weights={"a": 0.5, "b": 0.5},
        ),
    ):
        other = blend_strategies({"s": dropped}, predictions, features, era, FEATURE_SETS)["s"]
        assert not np.allclose(other, result)


def test_a_feature_set_name_neutralizes_against_that_sets_columns():
    predictions, features, era = _frames()
    by_name = _strategy(_spec("a"), _spec("b"), neutralization=Neutralization(0.6, "pair"))
    by_columns = _strategy(_spec("a"), _spec("b"), neutralization=Neutralization(0.6, ("f1", "f2")))

    result = blend_strategies({"n": by_name, "c": by_columns}, predictions, features, era, FEATURE_SETS)

    pd.testing.assert_series_equal(result["n"], result["c"], check_names=False, check_exact=True)


# --- a batch of strategies: what the harness asks for --------------------------------------


def _spy_on_projections(monkeypatch) -> list[tuple[list[str], list[str]]]:
    calls = []
    real = blend_module.era_feature_projection

    def spy(df, columns, neutralizers, **kwargs):
        calls.append((list(columns), list(neutralizers)))
        return real(df, columns, neutralizers, **kwargs)

    monkeypatch.setattr(blend_module, "era_feature_projection", spy)
    return calls


def test_a_sweep_of_proportions_solves_each_era_once_not_once_per_row(monkeypatch):
    """The reason the interface takes a mapping: rows sharing a column and a feature set share the solve (#35)."""
    predictions, features, era = _frames()
    calls = _spy_on_projections(monkeypatch)
    rows = {
        f"p{proportion:g}": _strategy(
            _spec("a"), _spec("b"), neutralization=Neutralization(proportion, tuple(FEATURES))
        )
        for proportion in (0.25, 0.5, 0.75, 1.0)
    }

    result = blend_strategies(rows, predictions, features, era, FEATURE_SETS)

    assert len(calls) == 1
    assert list(result.columns) == list(rows)
    assert not np.allclose(result["p0.25"], result["p1"])


def test_rows_with_different_weights_get_their_own_blend_but_share_the_feature_set(monkeypatch):
    predictions, features, era = _frames()
    calls = _spy_on_projections(monkeypatch)
    rows = {
        f"w{w:g}": _strategy(
            _spec("a"),
            _spec("b"),
            weights={"a": w, "b": 1 - w},
            neutralization=Neutralization(0.95, tuple(FEATURES)),
        )
        for w in (0.2, 0.5, 0.8)
    }

    blend_strategies(rows, predictions, features, era, FEATURE_SETS)

    # Three distinct blends, all projected onto one feature set: one grouped call, three columns.
    assert len(calls) == 1
    assert len(calls[0][0]) == 3


def test_scoring_strategies_together_matches_scoring_each_alone():
    predictions, features, era = _frames()
    rows = {
        "plain": _strategy(_spec("a"), _spec("b")),
        "blend_only": _strategy(_spec("a"), _spec("b"), neutralization=Neutralization(0.95, "all4")),
        "per_model": _strategy(
            _spec("a"), _spec("b", Neutralization(0.5, "pair")), neutralization=Neutralization(0.5, "all4")
        ),
    }

    together = blend_strategies(rows, predictions, features, era, FEATURE_SETS)

    for label, strategy in rows.items():
        alone = blend_strategies({label: strategy}, predictions, features, era, FEATURE_SETS)[label]
        # Grouped solves may differ in the last bits from a one-column solve, never more.
        np.testing.assert_allclose(together[label], alone, rtol=0, atol=1e-12)


def test_a_sweep_may_score_a_subset_of_the_models_it_was_given():
    predictions, features, era = _frames()

    result = blend_strategies(
        {"only_a": _strategy(_spec("a"), neutralization=Neutralization(0.5, "pair"))},
        predictions,
        features,
        era,
        FEATURE_SETS,
    )

    expected = _neutralized(predictions["a"], features, era, ("f1", "f2"), 0.5)
    pd.testing.assert_series_equal(result["only_a"], expected, check_names=False, check_exact=True)


# --- refusals ------------------------------------------------------------------------------


def test_a_model_without_predictions_is_named():
    predictions, features, era = _frames()

    with pytest.raises(KeyError, match=r"model\(s\) \['ghost'\]"):
        blend_strategies({"s": _strategy(_spec("ghost"))}, predictions, features, era, FEATURE_SETS)


def test_a_neutralizer_column_missing_from_the_features_is_named():
    predictions, features, era = _frames()
    strategy = _strategy(_spec("a"), neutralization=Neutralization(0.5, ("f1", "not_loaded")))

    with pytest.raises(KeyError, match="not_loaded"):
        blend_strategies({"s": strategy}, predictions, features, era, FEATURE_SETS)


def test_an_unresolved_feature_set_name_is_named():
    predictions, features, era = _frames()
    strategy = _strategy(_spec("a"), neutralization=Neutralization(0.5, "medium"))

    with pytest.raises(KeyError, match="'medium'"):
        blend_strategies({"s": strategy}, predictions, features, era, FEATURE_SETS)


def test_misaligned_frames_are_refused_not_silently_realigned():
    predictions, features, era = _frames()

    with pytest.raises(ValueError, match="`features` must share"):
        blend_strategies(
            {"s": _strategy(_spec("a"))}, predictions, features.iloc[::-1], era, FEATURE_SETS
        )
