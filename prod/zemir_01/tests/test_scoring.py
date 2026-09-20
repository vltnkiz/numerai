import pandas as pd
import pytest

from zemir.scoring import PredictionSanityError, rank_normalize, validate_predictions


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
