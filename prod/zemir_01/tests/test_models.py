"""`zemir.models.FittedModel`: how a trained estimator is fed, and what it can explain (issue #86).

Every trainer is seeded, so what these pin is output, not implementation: the
dtype and layout an estimator is handed, that row-chunked prediction is the
same prediction as one shot, and that the one feature scaling in the codebase
is applied identically at fit and predict.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from zemir import models
from zemir.models import FittedModel, train_linear, train_ridge, train_sgd, train_xgboost

from factories import FEATURE_COLUMNS, PredictColumn, make_dataset


class Spy:
    """An estimator that records every array it is handed."""

    def __init__(self) -> None:
        self.fed: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.fed.append(X)
        return X[:, 0].astype(np.float64)


def frame_and_array(n_rows: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    frame = make_dataset(rows_per_era=n_rows).train[FEATURE_COLUMNS].iloc[:n_rows]
    return frame, frame.to_numpy()


def fit_data():
    dataset = make_dataset(n_train_eras=6, rows_per_era=20, noise_scale=0.5)
    train = dataset.train
    return train[FEATURE_COLUMNS], train["target"], train["era"], dataset.validation


# --- how an estimator is fed ------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8])
def test_the_estimator_is_fed_the_recorded_dtype_whatever_the_frame_holds(dtype):
    frame, _ = frame_and_array()
    spy = Spy()

    FittedModel(spy, tuple(FEATURE_COLUMNS), dtype=dtype).predict(frame)

    assert spy.fed[0].dtype == dtype


def test_a_matching_dtype_is_passed_through_without_a_copy():
    # The booster's whole predict story: int8 goes to `inplace_predict` as it
    # is, so no float copy of the design matrix is ever made.
    frame, array = frame_and_array()
    spy = Spy()

    FittedModel(spy, tuple(FEATURE_COLUMNS), dtype=np.int8).predict(frame)

    assert np.shares_memory(spy.fed[0], array)


def test_order_c_feeds_the_estimator_a_row_major_matrix_and_the_default_keeps_the_frames_layout():
    frame, _ = frame_and_array()
    row_major, as_is = Spy(), Spy()

    FittedModel(row_major, tuple(FEATURE_COLUMNS), dtype=np.float64, order="C").predict(frame)
    FittedModel(as_is, tuple(FEATURE_COLUMNS), dtype=np.float64).predict(frame)

    # A frame's int8 block is column-major, and a plain cast keeps that.
    assert row_major.fed[0].flags.c_contiguous
    assert as_is.fed[0].flags.f_contiguous and not as_is.fed[0].flags.c_contiguous


def test_a_transform_never_sees_the_callers_own_array():
    # `_scale_features` works in place. Even when the dtype already matches, the
    # transform must be handed a copy — or predicting would rewrite the frame.
    frame, _ = frame_and_array()
    before = frame.copy()

    def shift_in_place(rows: np.ndarray) -> np.ndarray:
        rows -= 1
        return rows

    spy = Spy()
    FittedModel(spy, tuple(FEATURE_COLUMNS), dtype=np.int8, transform=shift_in_place).predict(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert (spy.fed[0] == before.to_numpy() - 1).all()


def test_predict_takes_its_columns_from_any_frame_that_has_them():
    frame, _ = frame_and_array()
    model = FittedModel(Spy(), ("feature_b", "feature_a"), dtype=np.int8)

    wide = frame.assign(unrelated=1)[["unrelated", "feature_a", "feature_c", "feature_b"]]
    model.predict(wide)

    # Handed feature_b then feature_a — the order it was fitted in, not the frame's.
    fed = model.estimator.fed[0]
    assert (fed[:, 0] == frame["feature_b"].to_numpy()).all()
    assert (fed[:, 1] == frame["feature_a"].to_numpy()).all()
    assert fed.shape[1] == 2


def test_predict_on_a_frame_missing_a_fitted_column_fails_loudly():
    frame, _ = frame_and_array()
    model = FittedModel(Spy(), ("feature_a", "not_there"), dtype=np.int8)

    with pytest.raises(KeyError, match="not_there"):
        model.predict(frame)


# --- each trainer records how it must be fed ---------------------------------------------


def test_each_trainer_records_the_dtype_layout_and_chunking_its_numbers_were_made_with():
    X, y, era, _ = fit_data()
    columns = tuple(FEATURE_COLUMNS)

    ols = train_linear(X, y)
    ridge = train_ridge(X, y, alpha=1.0)
    sgd = train_sgd(X, y, era, chunk_rows=25, max_epochs=1)
    booster = train_xgboost(X, y, era, trees_per_step=2, num_iters=1, batch_rows=25)

    # OLS: float64, row-major (the layout numpy's own promotion gave its
    # predictions), and single-shot — chunking moves its last bits.
    assert (ols.dtype, ols.order, ols.chunk_rows, ols.transform) == (np.float64, "C", None, None)
    assert (ridge.dtype, ridge.order, ridge.chunk_rows, ridge.transform) == (np.float32, "K", None, None)
    assert (sgd.dtype, sgd.order, sgd.chunk_rows) == (np.float32, "K", 25)
    assert sgd.transform is models._scale_features
    assert (booster.dtype, booster.order, booster.chunk_rows, booster.transform) == (np.int8, "K", 25, None)
    assert {m.columns for m in (ols, ridge, sgd, booster)} == {columns}


def test_ols_predicts_exactly_what_the_bare_estimator_did_before_fitted_model():
    # Before #86 `train_linear` returned a raw `LinearRegression` fitted on the
    # frame and predicted on it. Same coefficients, same predictions, to the bit,
    # and no feature-names warning now that it predicts from an array.
    X, y, _, validation = fit_data()
    bare = LinearRegression().fit(X, y)

    fitted = train_linear(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predicted = fitted.predict(validation)

    assert np.array_equal(fitted.estimator.coef_, bare.coef_)
    assert fitted.estimator.intercept_ == bare.intercept_
    assert predicted.dtype == np.float64
    assert np.array_equal(predicted, bare.predict(validation[FEATURE_COLUMNS]))


# --- chunked prediction -----------------------------------------------------------------


def chunk_sizes(n: int) -> list[int]:
    return [1, 3, 7, n - 1, n, n + 1, 10 * n]


def test_a_booster_predicts_bit_identically_at_every_chunk_size():
    # Each row's prediction depends only on that row, so chunking is exact here.
    X, y, era, validation = fit_data()
    booster = train_xgboost(X, y, era, trees_per_step=5, num_iters=1, batch_rows=1000)
    n = len(validation)

    one_shot = FittedModel(booster.estimator, booster.columns, dtype=np.int8).predict(validation)

    for chunk_rows in chunk_sizes(n):
        chunked = FittedModel(
            booster.estimator, booster.columns, dtype=np.int8, chunk_rows=chunk_rows
        ).predict(validation)
        assert np.array_equal(chunked, one_shot), chunk_rows


@pytest.mark.parametrize("trainer", ["ridge", "sgd"])
def test_a_linear_model_predicts_the_same_at_every_chunk_size(trainer):
    # Not bit-identical: BLAS sums a different number of rows per call, so the
    # last float32 bits move with the chunk size. Off-by-one at a chunk boundary
    # (a dropped or repeated row) would be a gross difference, not a rounding one.
    X, y, era, validation = fit_data()
    fitted = (
        train_ridge(X, y, alpha=1.0)
        if trainer == "ridge"
        else train_sgd(X, y, era, chunk_rows=25, max_epochs=1)
    )
    n = len(validation)

    def with_chunks(chunk_rows):
        return FittedModel(
            fitted.estimator,
            fitted.columns,
            dtype=fitted.dtype,
            chunk_rows=chunk_rows,
            transform=fitted.transform,
        )

    one_shot = with_chunks(None).predict(validation)
    for chunk_rows in chunk_sizes(n):
        chunked = with_chunks(chunk_rows).predict(validation)
        assert chunked.shape == one_shot.shape
        np.testing.assert_allclose(chunked, one_shot, rtol=1e-5, atol=1e-6, err_msg=str(chunk_rows))


def test_a_chunked_predict_is_fed_slices_of_the_requested_size():
    frame, _ = frame_and_array(10)
    spy = Spy()

    FittedModel(spy, tuple(FEATURE_COLUMNS), dtype=np.int8, chunk_rows=4).predict(frame)

    assert [len(chunk) for chunk in spy.fed] == [4, 4, 2]


# --- the one feature scaling ------------------------------------------------------------


def test_sgd_predicts_on_the_scaled_features_it_was_fitted_on():
    X, y, era, validation = fit_data()

    fitted = train_sgd(X, y, era, chunk_rows=25, max_epochs=2)

    scaled = (validation[FEATURE_COLUMNS].to_numpy().astype(np.float32) - 2) / 2
    np.testing.assert_allclose(fitted.predict(validation), fitted.estimator.predict(scaled), rtol=1e-5)


def test_sgd_scaling_is_one_rule_applied_at_fit_and_at_predict(monkeypatch):
    # Not "two call sites that agree": swap the rule for a different one and both
    # the fit and the predict follow it, because both go through the same method.
    X, y, era, validation = fit_data()
    applied: list[int] = []

    def shifted(rows: np.ndarray) -> np.ndarray:
        applied.append(len(rows))
        rows -= 4.0
        return rows

    monkeypatch.setattr(models, "_scale_features", shifted)
    fitted = train_sgd(X, y, era, chunk_rows=25, max_epochs=1)
    applied_while_fitting = len(applied)

    predicted = fitted.predict(validation)

    assert applied_while_fitting > 0
    assert len(applied) > applied_while_fitting
    unscaled = validation[FEATURE_COLUMNS].to_numpy().astype(np.float32)
    np.testing.assert_allclose(predicted, fitted.estimator.predict(unscaled - 4.0), rtol=1e-5)


# --- explain ----------------------------------------------------------------------------


@pytest.mark.parametrize("trainer", ["ols", "ridge", "sgd"])
def test_a_linear_models_explanation_is_its_coefficients_by_column_name(trainer):
    X, y, era, _ = fit_data()
    fitted = {
        "ols": lambda: train_linear(X, y),
        "ridge": lambda: train_ridge(X, y, alpha=1.0),
        "sgd": lambda: train_sgd(X, y, era, chunk_rows=25, max_epochs=1),
    }[trainer]()

    explanation = fitted.explain()

    assert explanation.index.name == "feature"
    assert explanation.index.tolist() == FEATURE_COLUMNS
    assert explanation.columns.tolist() == ["coef"]
    assert explanation["coef"].dtype == np.float64
    # In the space the estimator was fitted in: sgd is not rescaled, ridge's
    # float32 coefficients are upcast losslessly.
    assert np.array_equal(explanation["coef"].to_numpy(), fitted.estimator.coef_.astype(np.float64))


def booster_where_only_the_second_column_matters():
    """A booster over columns whose names are shuffled against their positions, one never split on."""
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame(
        {
            "c": rng.integers(0, 5, n).astype("int8"),
            "a": rng.integers(0, 5, n).astype("int8"),  # position 1: the signal
            "b": rng.integers(0, 5, n).astype("int8"),
            "d": np.full(n, 2, dtype="int8"),  # constant: no split can use it
        }
    )
    y = pd.Series(X["a"].astype(float) + rng.normal(scale=0.1, size=n))
    era = pd.Series([f"{1 + i // 40:04d}" for i in range(n)])
    return train_xgboost(
        X, y, era, trees_per_step=10, num_iters=0, max_depth=3, colsample_bytree=1.0, batch_rows=100
    )


def test_a_boosters_explanation_maps_its_positional_features_back_to_names():
    fitted = booster_where_only_the_second_column_matters()

    explanation = fitted.explain()

    assert explanation.columns.tolist() == ["weight", "gain", "cover"]
    assert explanation.index.tolist() == ["c", "a", "b", "d"]
    assert explanation["gain"].idxmax() == "a"
    # Cross-checked against the booster itself: position 1 is column "a".
    raw = fitted.estimator.get_score(importance_type="weight")
    assert explanation.loc["a", "weight"] == raw["f1"]
    assert explanation.loc["b", "weight"] == raw.get("f2", 0)


def test_a_feature_the_booster_never_split_on_has_no_weight_and_no_defined_gain():
    explanation = booster_where_only_the_second_column_matters().explain()

    assert explanation.loc["d", "weight"] == 0
    assert np.isnan(explanation.loc["d", "gain"])
    assert np.isnan(explanation.loc["d", "cover"])
    assert not explanation.loc["a"].isna().any()


def test_explaining_does_not_touch_the_booster():
    fitted = booster_where_only_the_second_column_matters()
    before = bytes(fitted.estimator.save_raw("ubj"))

    fitted.explain()

    assert fitted.estimator.feature_names is None
    assert bytes(fitted.estimator.save_raw("ubj")) == before


def test_a_boosters_explanation_carries_no_total_columns():
    # `total_gain = weight * gain`; left out because it is derivable and the
    # worst-reproducing of the five importance types (issue #83).
    assert not any(c.startswith("total") for c in booster_where_only_the_second_column_matters().explain())


def test_a_booster_whose_features_were_named_fails_rather_than_reading_zeros():
    fitted = booster_where_only_the_second_column_matters()
    fitted.estimator.feature_names = ["c", "a", "b", "d"]  # keys become real names

    with pytest.raises(ValueError, match="not a position"):
        fitted.explain()


def test_an_estimator_that_cannot_explain_raises_naming_its_type():
    # An empty frame would read as "the model uses nothing".
    model = FittedModel(PredictColumn(0), tuple(FEATURE_COLUMNS), dtype=np.float64)

    with pytest.raises(NotImplementedError, match="PredictColumn"):
        model.explain()


# --- save / load ------------------------------------------------------------------------


def every_trainer():
    X, y, era, validation = fit_data()
    fitted = {
        "ols": train_linear(X, y),
        "ridge": train_ridge(X, y, alpha=1.0),
        "sgd": train_sgd(X, y, era, chunk_rows=25, max_epochs=2),
        "booster": train_xgboost(X, y, era, trees_per_step=3, num_iters=1, batch_rows=25),
    }
    return fitted, validation


@pytest.mark.parametrize("trainer", ["ols", "ridge", "sgd", "booster"])
def test_a_reloaded_model_predicts_and_explains_bit_identically(trainer, tmp_path):
    # Bit-identical, not close: every trainer is seeded, and what is being
    # pinned is that nothing about how the estimator is fed (dtype, layout,
    # chunking, sgd's scaling) is lost on the way through the disk.
    fitted, validation = every_trainer()
    original = fitted[trainer]

    original.save(tmp_path / "model")
    reloaded = FittedModel.load(tmp_path / "model")

    assert (reloaded.columns, reloaded.dtype, reloaded.order, reloaded.chunk_rows) == (
        original.columns,
        np.dtype(original.dtype),
        original.order,
        original.chunk_rows,
    )
    assert reloaded.transform is original.transform
    np.testing.assert_array_equal(reloaded.predict(validation), original.predict(validation))
    pd.testing.assert_frame_equal(reloaded.explain(), original.explain())


def test_a_reloaded_booster_explains_without_the_booster_carrying_names(tmp_path):
    fitted, _ = every_trainer()
    fitted["booster"].save(tmp_path)

    reloaded = FittedModel.load(tmp_path)

    assert reloaded.estimator.feature_names is None
    assert list(reloaded.explain().index) == FEATURE_COLUMNS


def test_a_saved_model_holds_no_pickle(tmp_path):
    # A pickled sklearn estimator is readable only by the version that wrote it.
    fitted, _ = every_trainer()
    for name in ("ols", "sgd", "booster"):
        fitted[name].save(tmp_path / name)

    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files
    for path in files:
        assert path.suffix in {".json", ".ubj", ".npz"}
        assert not path.read_bytes().startswith(b"\x80")  # a pickle's protocol marker


def test_an_estimator_it_cannot_write_faithfully_raises_before_writing_anything(tmp_path):
    model = FittedModel(PredictColumn(0), tuple(FEATURE_COLUMNS), dtype=np.float64)

    with pytest.raises(TypeError, match="PredictColumn"):
        model.save(tmp_path / "model")

    assert not (tmp_path / "model").exists()


def test_an_unregistered_transform_raises_rather_than_saving_a_model_that_predicts_differently(
    tmp_path,
):
    fitted, _ = every_trainer()
    model = FittedModel(
        fitted["ols"].estimator, fitted["ols"].columns, dtype=np.float64, transform=lambda X: X
    )

    with pytest.raises(TypeError, match="not registered"):
        model.save(tmp_path / "model")

    assert not (tmp_path / "model").exists()


def test_a_directory_with_no_meta_is_an_interrupted_save_and_does_not_load(tmp_path):
    fitted, _ = every_trainer()
    fitted["ols"].save(tmp_path)
    (tmp_path / "meta.json").unlink()

    with pytest.raises(FileNotFoundError, match="meta.json"):
        FittedModel.load(tmp_path)


def test_a_format_it_does_not_know_fails_naming_both_versions(tmp_path):
    fitted, _ = every_trainer()
    fitted["ols"].save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    (tmp_path / "meta.json").write_text(json.dumps({**meta, "format": 2}))

    with pytest.raises(ValueError, match="format 2.*reads format 1"):
        FittedModel.load(tmp_path)


def test_a_missing_estimator_file_fails_rather_than_returning_an_empty_model(tmp_path):
    fitted, _ = every_trainer()
    fitted["booster"].save(tmp_path)
    (tmp_path / "booster.ubj").unlink()

    with pytest.raises(Exception):
        FittedModel.load(tmp_path)


def test_coefficients_that_do_not_match_the_columns_fail_loudly(tmp_path):
    fitted, _ = every_trainer()
    fitted["ols"].save(tmp_path)
    np.savez(tmp_path / "linear.npz", coef=np.zeros(2), intercept=np.zeros(()))

    with pytest.raises(ValueError, match="coefficients"):
        FittedModel.load(tmp_path)
