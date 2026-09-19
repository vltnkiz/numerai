from dataclasses import replace

from zemir.config import LIVE
from zemir.harness import fit_validation_predictions
from zemir.strategy import BlendSpec, ModelSpec, Strategy

from factories import FEATURE_COLUMNS, make_dataset

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
        feature_columns=FEATURE_COLUMNS,
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
        feature_columns=FEATURE_COLUMNS,
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
        feature_columns=FEATURE_COLUMNS,
        load_train=load_train,
        load_validation=load_validation,
        harness_dir=tmp_path,
    )

    assert result.predictions["era"].nunique() == 3
