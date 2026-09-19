import pandas as pd
import pytest

from zemir.scoring import PredictionSanityError, neutralize_predictions, rank_normalize, validate_predictions


def test_neutralize_predictions_zero_proportion_is_a_no_op():
    df = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0001"],
            "prediction": [0.1, 0.4, 0.6, 0.9],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    neutralized = neutralize_predictions(df, ["f1"], proportion=0.0)

    pd.testing.assert_series_equal(
        neutralized, df["prediction"].rename("prediction_neutralized")
    )


def test_neutralize_predictions_full_proportion_removes_a_perfectly_linear_exposure():
    # `prediction` is exactly `f1`, so a full-proportion projection onto `f1`
    # explains all of it — the residual should be ~0 everywhere.
    df = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0001"],
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    neutralized = neutralize_predictions(df, ["f1"], proportion=1.0)

    assert neutralized.abs().max() < 1e-8


def test_neutralize_predictions_neutralizes_each_era_independently():
    # Era "0001" is perfectly explained by f1 (residual ~0 at full proportion).
    # Era "0002" has a *constant* f1, so nothing there is explained by it —
    # each row's residual is just its own deviation from the era's mean
    # prediction (6.0), regardless of era "0001"'s own f1 relationship.
    df = pd.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "prediction": [1.0, 2.0, 5.0, 7.0],
            "f1": [1.0, 2.0, 3.0, 3.0],
        }
    )

    neutralized = neutralize_predictions(df, ["f1"], proportion=1.0)

    era_0001 = neutralized[df["era"] == "0001"]
    assert era_0001.abs().max() < 1e-8

    era_0002 = neutralized[df["era"] == "0002"]
    assert era_0002.iloc[0] == pytest.approx(-1.0)
    assert era_0002.iloc[1] == pytest.approx(1.0)


def test_rank_normalize_maps_into_the_open_unit_interval_preserving_order():
    values = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0], index=["a", "b", "c", "d", "e"])

    normalized = rank_normalize(values)

    assert normalized.min() > 0.0
    assert normalized.max() < 1.0
    assert list(normalized.sort_values().index) == list(values.sort_values().index)


def test_validate_predictions_accepts_well_formed_predictions():
    predictions = pd.Series([0.1, 0.5, 0.9, 0.3])
    era = pd.Series(["0001", "0001", "0002", "0002"])

    validate_predictions(predictions, era)  # must not raise


def test_validate_predictions_rejects_nulls():
    predictions = pd.Series([0.1, None, 0.9])
    era = pd.Series(["0001", "0001", "0001"])

    with pytest.raises(PredictionSanityError, match="null prediction"):
        validate_predictions(predictions, era)


def test_validate_predictions_rejects_values_outside_open_unit_interval():
    predictions = pd.Series([0.1, 0.5, 1.0])
    era = pd.Series(["0001", "0001", "0001"])

    with pytest.raises(PredictionSanityError, match=r"outside \(0, 1\)"):
        validate_predictions(predictions, era)


def test_validate_predictions_rejects_a_zero_variance_era():
    predictions = pd.Series([0.5, 0.5, 0.5, 0.2, 0.8])
    era = pd.Series(["0001", "0001", "0001", "0002", "0002"])

    with pytest.raises(PredictionSanityError, match=r"zero-variance predictions in era\(s\) \['0001'\]"):
        validate_predictions(predictions, era)
