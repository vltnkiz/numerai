"""Training stage: fit a Model on train.parquet, score it on validation.parquet.

Ties the Model protocol (zemir/models/base.py) to zemir/metrics.py's
validation scoring — per decision #1, a validation score must be computed
before predictions are allowed to reach the submission stage. Which model
type runs and which features it sees are both caller-supplied, so this stage
is identical for linear regression and, later, era-boosted XGBoost.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zemir.metrics import ValidationScore, score_validation
from zemir.models.base import FitResult, Model


@dataclass
class TrainResult:
    fit_result: FitResult
    validation_predictions: pd.Series
    validation_score: ValidationScore


def train_and_validate(
    model: Model,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_col: str = "target",
    era_col: str = "era",
) -> TrainResult:
    """Fit `model` on `train_df` and score it against `validation_df`.

    Rows with a null target are dropped from both frames first (validation
    has some eras with no target yet, per era-boosting's notebook).
    """
    train_df = train_df.dropna(subset=[target_col])
    validation_df = validation_df.dropna(subset=[target_col])

    fit_result = model.fit(
        train_df[feature_columns],
        train_df[target_col],
        train_df[era_col],
    )

    predictions = pd.Series(
        model.predict(validation_df[feature_columns]),
        index=validation_df.index,
        name="prediction",
    )

    scored = validation_df[[era_col, target_col]].copy()
    scored["prediction"] = predictions
    validation_score = score_validation(
        scored, target_col=target_col, prediction_col="prediction", era_col=era_col
    )

    return TrainResult(
        fit_result=fit_result,
        validation_predictions=predictions,
        validation_score=validation_score,
    )
