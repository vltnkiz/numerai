import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from zemir.config import LIVE, PRODUCTION_STRATEGY
from zemir.harness import (
    PREDICTIONS_FILENAME,
    NoPersistedModels,
    UnscoreableStrategy,
    explain_models,
    fit_validation_predictions,
    load_fit_record,
    load_models,
    rank_feature_exposure,
    score_configs,
    unscoreable_reason,
)
from zemir.strategy import BlendSpec, ModelSpec, Neutralization, Strategy

from factories import FEATURE_SETS, make_dataset

# `ols` is a real trainer (unlike a fake `predict_column_trainer`) — sklearn's
# LinearRegression fits the ~200-row synthetic dataset instantly, so there's
# no speed reason left to fake it now that `fit_validation_predictions` takes
# a real `Strategy` rather than an injectable `build_trainers` callable.
STRATEGY = Strategy(
    models=(ModelSpec(name="linear", features="medium", trainer="ols"),),
    blend=BlendSpec(),
)


def _loaders(dataset):
    """Loaders that record call order, standing in for a real `load_split` pair."""
    calls: list[str] = []

    def load_train():
        calls.append("train")
        return dataset.train

    def load_validation():
        calls.append("validation")
        return dataset.validation

    return load_train, load_validation, calls


def test_fit_validation_predictions_writes_the_cache_without_touching_the_network(tmp_path):
    dataset = make_dataset()
    load_train, load_validation, _ = _loaders(dataset)

    result = fit_validation_predictions(
        LIVE,
        STRATEGY,
        run_id="test-fit",
        feature_sets=FEATURE_SETS,
        load_train=load_train,
        load_validation=load_validation,
        harness_dir=tmp_path,
    )

    assert result.predictions_path.exists()
    assert (result.run_dir / "fit_config.json").exists()
    assert set(result.predictions.columns) >= {"era", "target", "linear"}


def test_fit_validation_predictions_calls_load_train_before_load_validation(tmp_path):
    # The one-split-at-a-time discipline (issue #47/#51) lives in *when* this
    # calls each loader, not in what it's handed — this is what proves it's
    # preserved: train is fully consumed (fit, then freed) before validation
    # is ever loaded.
    dataset = make_dataset()
    load_train, load_validation, calls = _loaders(dataset)

    fit_validation_predictions(
        LIVE,
        STRATEGY,
        run_id="ordering",
        feature_sets=FEATURE_SETS,
        load_train=load_train,
        load_validation=load_validation,
        harness_dir=tmp_path,
    )

    assert calls == ["train", "validation"]


def test_fit_validation_predictions_respects_max_eras(tmp_path):
    dataset = make_dataset(n_train_eras=10, n_validation_eras=5, rows_per_era=20)
    load_train, load_validation, _ = _loaders(dataset)
    config = replace(LIVE, max_eras=3)

    result = fit_validation_predictions(
        config,
        STRATEGY,
        run_id="max-eras",
        feature_sets=FEATURE_SETS,
        load_train=load_train,
        load_validation=load_validation,
        harness_dir=tmp_path,
    )

    assert result.predictions["era"].nunique() == 3


def test_fit_validation_predictions_fits_two_specs_at_different_widths(tmp_path):
    dataset = make_dataset()
    load_train, load_validation, _ = _loaders(dataset)
    strategy = Strategy(
        models=(
            ModelSpec(name="linear_narrow", features="narrow", trainer="ols"),
            ModelSpec(name="linear_wide", features="medium", trainer="ols"),
        ),
        blend=BlendSpec(),
    )

    result = fit_validation_predictions(
        LIVE,
        strategy,
        run_id="two-widths",
        feature_sets=FEATURE_SETS,
        load_train=load_train,
        load_validation=load_validation,
        harness_dir=tmp_path,
    )

    assert set(result.predictions.columns) >= {"linear_narrow", "linear_wide"}
    # The two fits learned different things: one saw a single column, one saw three.
    assert not result.predictions["linear_narrow"].equals(result.predictions["linear_wide"])


# --- persisted models (issue #87) -------------------------------------------------------

TWO_MODELS = Strategy(
    models=(
        ModelSpec(name="linear", features="medium", trainer="ols"),
        ModelSpec(
            name="era_boost",
            features="medium",
            trainer="xgboost",
            params={"num_iters": 1, "trees_per_step": 3, "batch_rows": 25},
        ),
    ),
    blend=BlendSpec(),
)


def _fit(tmp_path, strategy=TWO_MODELS, run_id="run"):
    dataset = make_dataset()
    load_train, load_validation, _ = _loaders(dataset)
    return dataset, fit_validation_predictions(
        LIVE,
        strategy,
        run_id=run_id,
        feature_sets=FEATURE_SETS,
        load_train=load_train,
        load_validation=load_validation,
        harness_dir=tmp_path,
    )


def test_each_fitted_model_is_kept_beside_the_predictions_and_predicts_what_the_cache_holds(
    tmp_path,
):
    dataset, result = _fit(tmp_path)

    models = load_models(result.run_dir)

    assert list(models) == ["linear", "era_boost"]  # the strategy's declared order
    for name, model in models.items():
        np.testing.assert_array_equal(
            model.predict(dataset.validation).astype("float32"),
            result.predictions[name].to_numpy(),
        )


def test_models_are_saved_before_validation_is_loaded(tmp_path):
    # The fit is the ~1h part: a validation load that fails must not cost it.
    dataset = make_dataset()
    load_train, _, _ = _loaders(dataset)

    def failing_validation():
        raise RuntimeError("validation would not load")

    with pytest.raises(RuntimeError, match="would not load"):
        fit_validation_predictions(
            LIVE,
            TWO_MODELS,
            run_id="dies",
            feature_sets=FEATURE_SETS,
            load_train=load_train,
            load_validation=failing_validation,
            harness_dir=tmp_path,
        )

    assert (tmp_path / "dies" / "models" / "linear" / "meta.json").exists()
    # ...and it is still not a cache, so `latest_cache` will not pick it up.
    assert not (tmp_path / "dies" / PREDICTIONS_FILENAME).exists()


def test_a_cache_from_before_models_were_kept_says_so_rather_than_crashing(tmp_path):
    _, result = _fit(tmp_path)
    shutil.rmtree(result.run_dir / "models")  # what every existing cache looks like

    with pytest.raises(NoPersistedModels, match="fitted before models were kept"):
        load_models(result.run_dir)
    with pytest.raises(NoPersistedModels):
        explain_models(result.run_dir)
    assert result.predictions_path.exists()  # and the cache itself is untouched


def test_a_model_the_cache_names_but_models_lacks_is_an_error_not_a_smaller_dict(tmp_path):
    _, result = _fit(tmp_path)
    shutil.rmtree(result.run_dir / "models" / "era_boost")

    with pytest.raises(FileNotFoundError, match="era_boost"):
        load_models(result.run_dir)


def test_a_corrupt_model_fails_loudly_and_is_not_mistaken_for_an_old_cache(tmp_path):
    _, result = _fit(tmp_path)
    (result.run_dir / "models" / "linear" / "linear.npz").write_bytes(b"not an npz")

    with pytest.raises(Exception) as raised:
        load_models(result.run_dir)

    assert not isinstance(raised.value, NoPersistedModels)


def test_explain_ranks_a_linear_model_by_absolute_coefficient_and_a_booster_by_gain(tmp_path):
    _, result = _fit(tmp_path)

    explained = explain_models(result.run_dir)

    linear = explained["linear"]
    assert list(linear.columns) == ["coef"]
    assert linear["coef"].abs().is_monotonic_decreasing
    booster = explained["era_boost"]
    assert list(booster.columns) == ["weight", "gain", "cover"]
    gain = booster["gain"].dropna()
    assert gain.is_monotonic_decreasing
    # A feature it never split on has no gain, and sorts after every one that did.
    assert booster["gain"].isna().to_numpy()[len(gain) :].all()


# --- scoring a cache under strategies (issue #88) -----------------------------------------
#
# `score_configs` reads validation features and the meta model from disk, so those three
# reads (and only those) are swapped for the synthetic dataset. Everything downstream of
# them — the blend, the fit-field check, the paid metrics, what is written — runs for real,
# against a cache fitted by the real `ols` and `xgboost` trainers (`TWO_MODELS`).


@pytest.fixture
def scoring_env(monkeypatch):
    dataset = make_dataset()
    validation = dataset.validation
    rng = np.random.default_rng(11)
    meta_model = pd.Series(rng.normal(size=len(validation)), index=validation.index)

    def load_features(version, columns, *, eras=None, **_):
        frame = validation[["era", *columns]]
        return frame if eras is None else frame[frame["era"].isin(eras)]

    monkeypatch.setattr(
        "zemir.harness.resolve_feature_sets",
        lambda version, names, **_: {n: FEATURE_SETS[n] for n in dict.fromkeys(names)},
    )
    monkeypatch.setattr("zemir.harness.load_validation_features", load_features)
    monkeypatch.setattr("zemir.harness.load_meta_model", lambda version, **_: meta_model)


def _blend(weights=None, neutralization=None):
    return BlendSpec(weights=weights, neutralization=neutralization)


def _row(strategy=TWO_MODELS, **blend):
    return replace(strategy, blend=_blend(**blend))


def test_score_configs_scores_each_strategy_as_a_row_and_records_what_it_scored(tmp_path, scoring_env):
    _, fitted = _fit(tmp_path)
    rows = {
        "raw": _row(),
        "neutralized": _row(neutralization=Neutralization(0.95, "medium")),
    }

    result = score_configs(rows, run_dir=fitted.run_dir)

    assert list(result.summary.index) == ["raw", "neutralized"]
    assert (fitted.run_dir / "scores.csv").exists()
    written = json.loads((fitted.run_dir / "scoring_strategies.json").read_text())
    assert written["fit_fields"] == "verified"
    assert set(written["strategies"]) == {"raw", "neutralized"}
    assert written["strategies"]["neutralized"]["blend"]["neutralization"] == {
        "proportion": 0.95,
        "features": "medium",
    }


def test_score_configs_applies_a_per_model_neutralization_to_the_cached_predictions(tmp_path, scoring_env):
    """The harness half of #88: a neutralization on one model is scored, not ignored."""
    _, fitted = _fit(tmp_path)
    linear, era_boost = TWO_MODELS.models
    per_model = replace(
        TWO_MODELS, models=(linear, replace(era_boost, neutralization=Neutralization(1.0, "narrow")))
    )

    result = score_configs({"plain": _row(), "per_model": per_model}, run_dir=fitted.run_dir)

    assert result.summary.loc["plain", "mean_corr"] != result.summary.loc["per_model", "mean_corr"]


def test_score_configs_refuses_a_strategy_fitted_differently_from_the_cache(tmp_path, scoring_env):
    _, fitted = _fit(tmp_path)
    linear, era_boost = TWO_MODELS.models
    deeper = replace(era_boost, params={**era_boost.params, "num_iters": 2})
    ridge = replace(linear, trainer="ridge", params={"alpha": 1.0})

    with pytest.raises(UnscoreableStrategy, match=r"other: .*era_boost\.params"):
        score_configs({"other": replace(TWO_MODELS, models=(linear, deeper))}, run_dir=fitted.run_dir)
    with pytest.raises(UnscoreableStrategy, match=r"linear\.trainer"):
        score_configs({"other": replace(TWO_MODELS, models=(ridge, era_boost))}, run_dir=fitted.run_dir)


def test_score_configs_refuses_a_model_the_cache_never_fitted(tmp_path, scoring_env):
    _, fitted = _fit(tmp_path)
    ghost = Strategy(models=(ModelSpec(name="ghost", features="medium", trainer="ols"),), blend=_blend())

    with pytest.raises(UnscoreableStrategy, match=r"\['ghost'\] were not fitted"):
        score_configs({"g": ghost}, run_dir=fitted.run_dir)


def test_a_sweep_may_score_a_subset_of_a_caches_models_and_any_weights(tmp_path, scoring_env):
    _, fitted = _fit(tmp_path)
    linear, _ = TWO_MODELS.models
    rows = {
        "linear_only": Strategy(models=(linear,), blend=_blend()),
        "weighted": _row(weights={"linear": 0.2, "era_boost": 0.8}),
    }

    result = score_configs(rows, run_dir=fitted.run_dir)

    assert list(result.summary.index) == ["linear_only", "weighted"]


def _make_legacy(run_dir, models=("era_boost", "linear")):
    """Rewrite `fit_config.json` the way every cache fitted before #74 looks: model names, no strategy."""
    path = run_dir / "fit_config.json"
    fit = json.loads(path.read_text())
    del fit["strategy"]
    fit["models"] = list(models)
    path.write_text(json.dumps(fit))


def test_a_cache_with_no_recorded_strategy_is_scored_by_model_name_and_labelled_unverified(
    tmp_path, scoring_env
):
    _, fitted = _fit(tmp_path)
    _make_legacy(fitted.run_dir)
    record = load_fit_record(fitted.run_dir)
    assert not record.verified
    assert [spec.name for spec in record.strategy.models] == ["linear", "era_boost"]  # STRATEGIES order

    # Fit fields cannot be held to a record that does not exist — even ones that differ from what was fitted.
    ridge = ModelSpec(name="linear", features="medium", trainer="ridge", params={"alpha": 1.0})
    result = score_configs({"row": Strategy(models=(ridge,), blend=_blend())}, run_dir=fitted.run_dir)

    assert list(result.summary.index) == ["row"]
    written = json.loads((fitted.run_dir / "scoring_strategies.json").read_text())
    assert written["fit_fields"].startswith("unverified")


def test_a_record_from_before_neutralization_existed_still_holds_its_fit_fields(tmp_path, scoring_env):
    """Its blend is in the old vocabulary, but which models were fitted, and how, is not."""
    _, fitted = _fit(tmp_path)
    path = fitted.run_dir / "fit_config.json"
    fit = json.loads(path.read_text())
    fit["strategy"]["blend"] = {"weights": None, "neutralize": None, "proportion": 0.95}
    for model in fit["strategy"]["models"]:
        model["neutralize"] = model.pop("neutralization")
    path.write_text(json.dumps(fit))

    record = load_fit_record(fitted.run_dir)

    assert record.verified
    assert [spec.name for spec in record.strategy.models] == ["linear", "era_boost"]
    deeper = replace(TWO_MODELS.models[1], params={"num_iters": 2})
    assert unscoreable_reason(replace(TWO_MODELS, models=(TWO_MODELS.models[0], deeper)), record)


def test_a_legacy_cache_naming_a_model_no_strategy_defines_says_so(tmp_path, scoring_env):
    _, fitted = _fit(tmp_path)
    _make_legacy(fitted.run_dir, models=("mystery",))

    with pytest.raises(KeyError, match="mystery"):
        load_fit_record(fitted.run_dir)


def test_production_is_not_expressible_against_a_cache_fitted_with_other_hyperparameters(
    tmp_path, scoring_env
):
    """The baseline row is `PRODUCTION_STRATEGY` or nothing: this cache's tiny booster is not production's."""
    _, fitted = _fit(tmp_path)

    reason = unscoreable_reason(PRODUCTION_STRATEGY, load_fit_record(fitted.run_dir))

    assert reason is not None
    assert "fitted differently" in reason
    assert "era_boost.params" in reason


def test_production_is_expressible_against_a_legacy_cache_that_holds_its_models(tmp_path, scoring_env):
    """Names only, and labelled as such — the shape of the cache `config.py` cites for re-measuring K."""
    _, fitted = _fit(tmp_path)
    _make_legacy(fitted.run_dir)

    record = load_fit_record(fitted.run_dir)

    assert unscoreable_reason(PRODUCTION_STRATEGY, record) is None
    assert not record.verified


def test_a_strategy_equal_in_fit_fields_to_the_cache_is_expressible_whatever_it_neutralizes(
    tmp_path, scoring_env
):
    _, fitted = _fit(tmp_path)
    swept = _row(
        weights={"linear": 0.3, "era_boost": 0.7},
        neutralization=Neutralization(0.5, ("feature_a",)),
    )

    assert unscoreable_reason(swept, load_fit_record(fitted.run_dir)) is None


def test_rank_feature_exposure_ranks_the_features_a_blend_leans_on(tmp_path, scoring_env):
    _, fitted = _fit(tmp_path)

    ranking = rank_feature_exposure(fitted.run_dir)

    assert ranking.is_monotonic_decreasing
    assert set(ranking.index) == set(FEATURE_SETS["medium"])
    # `target` is an affine function of feature_a, so both fitted models track it.
    assert ranking.index[0] == "feature_a"


def test_score_configs_hands_the_whole_sweep_to_the_shared_blend_in_one_call(
    tmp_path, scoring_env, monkeypatch
):
    """The other half of the pin in test_pipeline.py: one call, every row, so projections are shared."""
    import zemir.blend as blend_module
    import zemir.harness as harness_module

    _, fitted = _fit(tmp_path)
    calls = []
    real = blend_module.blend_strategies

    def spy(strategies, *args, **kwargs):
        calls.append(list(strategies))
        return real(strategies, *args, **kwargs)

    monkeypatch.setattr(harness_module, "blend_strategies", spy)
    rows = {
        f"p{p:g}": _row(neutralization=Neutralization(p, "medium")) for p in (0.25, 0.5, 0.75)
    }

    score_configs(rows, run_dir=fitted.run_dir)

    assert calls == [["p0.25", "p0.5", "p0.75"]]


def test_a_live_runs_submitted_blend_row_is_the_harness_row_for_the_same_strategy(tmp_path, scoring_env):
    """#93's destination, as one assertion: one scorer, one vocabulary, one number.

    The same strategy, fitted by the same real trainers on the same data, once
    by the harness and once by the live pipeline. The live run's
    `combined_neutralized` row and the harness's row for that strategy agree
    exactly in every column the live run records, and the gate's number is
    that row's `mean_corr`. The harness's one extra column is its ranking
    proxy, which a live run leaves out (docs/adr/0003).
    """
    from zemir.scoring import VALIDATION_PAYOUT_PROXY
    import zemir.harness as harness_module
    from zemir.pipeline import SUBMITTED_BLEND, run_pipeline, score_run

    strategy = replace(
        TWO_MODELS,
        models=(TWO_MODELS.models[0], replace(TWO_MODELS.models[1], neutralization=Neutralization(0.5, "narrow"))),
        blend=_blend(weights={"linear": 0.3, "era_boost": 0.7}, neutralization=Neutralization(0.95, "medium")),
    )
    _, fitted = _fit(tmp_path / "harness", strategy=strategy)
    harness = score_configs({"production": strategy}, run_dir=fitted.run_dir)

    live = run_pipeline(
        LIVE,
        run_id="live",
        strategy=strategy,
        feature_sets=FEATURE_SETS,
        dataset=make_dataset(),
        runs_dir=tmp_path / "live",
    )
    scores = score_run(
        live,
        meta_model=harness_module.load_meta_model(LIVE.data_version),
        scoring_universe=FEATURE_SETS[LIVE.feature_set],
    )

    # NaN would compare equal to NaN and prove nothing.
    assert harness.summary.loc["production"].notna().all()
    harness_row = harness.summary.loc["production"]
    assert list(harness_row.index) == [*scores.paid.summary.columns, VALIDATION_PAYOUT_PROXY]
    pd.testing.assert_series_equal(
        scores.paid.summary.loc[SUBMITTED_BLEND],
        harness_row.drop(VALIDATION_PAYOUT_PROXY),
        check_names=False,
    )
    assert live.gate_corr == harness.summary.loc["production", "mean_corr"]
