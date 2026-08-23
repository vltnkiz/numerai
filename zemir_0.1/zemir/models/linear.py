"""Linear regression Model implementation.

Wraps sklearn's LinearRegression to the Model protocol (zemir/models/base.py).
Ignores `era` and `checkpoint_path` — plain OLS has no per-era training
behavior and nothing to resume from; both params exist only so the pipeline
can call every model type identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from zemir.models.base import FitResult


class LinearRegressionModel:
    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        *,
        checkpoint_path: Path | None = None,
    ) -> FitResult:
        self._model.fit(X, y)
        return FitResult(model=self._model, history=None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
