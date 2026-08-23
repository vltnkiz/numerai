"""Model protocol shared by every model type in the pipeline.

A structural `typing.Protocol`, not an ABC, per
docs/adr/0001-zemir-pipeline-architecture.md: linear regression and
era-boosted XGBoost share no base-class behavior, only this call shape. Every
model receives `era` and `checkpoint_path` even if it ignores them, so the
pipeline can call every model type identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model: object
    history: pd.DataFrame | None


@runtime_checkable
class Model(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        *,
        checkpoint_path: Path | None = None,
    ) -> FitResult: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
