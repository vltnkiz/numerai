"""Runtime sanity checks on final submitted predictions.

Per [Automated testing strategy](https://github.com/vltnkiz/numerai/issues/13):
no pytest/CI suite for code correctness (that stays manually smoke-tested),
but a hard-stop guard on ML output coherence before submission, same pattern
as `zemir.pipeline.ValidationScoreBelowThreshold`. Config validation is
explicitly out of scope.
"""

from __future__ import annotations

import pandas as pd


class PredictionSanityError(RuntimeError):
    """Raised when final predictions fail a sanity check before submission."""


def validate_no_nulls(predictions: pd.Series) -> None:
    if predictions.isna().any():
        n = int(predictions.isna().sum())
        raise PredictionSanityError(f"{n} null prediction(s) found — refusing to submit")


def validate_non_constant_per_era(predictions: pd.Series, era: pd.Series) -> None:
    variance_by_era = predictions.groupby(era).var()
    constant_eras = variance_by_era[variance_by_era.fillna(0) == 0].index.tolist()
    if constant_eras:
        raise PredictionSanityError(
            f"zero-variance predictions in era(s) {constant_eras} — refusing to submit"
        )


def validate_predictions(predictions: pd.Series, era: pd.Series) -> None:
    validate_no_nulls(predictions)
    validate_non_constant_per_era(predictions, era)
