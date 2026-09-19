"""Synthetic data for tests that must never touch the network.

`make_dataset` stands in for `zemir.data.download`'s `Dataset`: same shape
(`train`/`validation`/`live` frames plus `feature_columns`), fabricated in
memory so `run_pipeline` and `fit_validation_predictions`'s whole post-fit
path — combine, neutralize, rank-normalize, score, validate — runs in
milliseconds instead of behind an 8.4 GB download (issue #75).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zemir.data import Dataset
from zemir.strategy import ModelSpec

FEATURE_COLUMNS = ["feature_a", "feature_b", "feature_c"]
# Feature-set name -> columns, as `zemir.data.resolve_feature_sets` would build it.
# `narrow` is a strict subset of `medium`, so a strategy naming both has a union
# equal to `medium` — the shape `small` (42) inside `medium` (780) would have.
FEATURE_SETS = {"medium": FEATURE_COLUMNS, "narrow": ["feature_a"]}
SIGNAL_COLUMN = "feature_a"


def _split(*, n_eras: int, rows_per_era: int, start_era: int, seed: int) -> pd.DataFrame:
    """One split: `rows_per_era` rows in each of `n_eras` eras, features int8-like `{0..4}`."""
    rng = np.random.default_rng(seed)
    n = n_eras * rows_per_era
    eras = [f"{start_era + i:04d}" for i in range(n_eras) for _ in range(rows_per_era)]
    features = {col: rng.integers(0, 5, size=n).astype("int8") for col in FEATURE_COLUMNS}
    frame = pd.DataFrame({"era": eras, **features})
    frame.index = pd.Index([f"n{start_era:04d}{i:05d}" for i in range(n)], name="id")
    return frame


def make_dataset(
    *,
    n_train_eras: int = 10,
    n_validation_eras: int = 5,
    rows_per_era: int = 20,
    signal_column: str = SIGNAL_COLUMN,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> Dataset:
    """A ~200-row synthetic `Dataset`: `target` tracks `signal_column` plus noise.

    `noise_scale=0.0` (the default) makes `target` an exact affine function of
    `signal_column`, so a trainer that predicts that column scores close to
    the maximum possible correlation — useful for exercising the
    `MIN_VALIDATION_MEAN_CORR` gate from both sides. A larger `noise_scale`
    weakens that relationship instead of removing it outright.
    """
    train = _split(n_eras=n_train_eras, rows_per_era=rows_per_era, start_era=1, seed=seed)
    validation = _split(
        n_eras=n_validation_eras, rows_per_era=rows_per_era, start_era=1000, seed=seed + 1
    )
    live = _split(n_eras=1, rows_per_era=rows_per_era, start_era=2000, seed=seed + 2)

    rng = np.random.default_rng(seed + 3)
    for frame in (train, validation):
        noise = rng.normal(scale=noise_scale, size=len(frame)) if noise_scale else 0.0
        frame["target"] = frame[signal_column].astype(float) + noise

    return Dataset(train=train, validation=validation, live=live, feature_columns=FEATURE_COLUMNS)


class PredictColumn:
    """A fitted model whose prediction is just one input column, unchanged.

    Stands in for a real trainer's fitted model — deterministic, and fast
    enough that a test never needs sklearn/xgboost to exercise the pipeline's
    own logic (combine, neutralize, score) rather than a model's. Records the
    columns of the last frame it was asked to predict on, so a test can see
    exactly what a model was handed.
    """

    def __init__(self, column: str, sign: float = 1.0) -> None:
        self.column = column
        self.sign = sign
        self.predicted_on: list[str] | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predicted_on = list(X.columns)
        return self.sign * X[self.column].to_numpy(dtype=float)


def fit_predict_column(X, y, era, *, column: str, **_: object) -> PredictColumn:
    """A trainer (registered as `predict_column`) that fits nothing and predicts `column`."""
    return PredictColumn(column)


def fit_predict_negated_column(X, y, era, *, column: str, **_: object) -> PredictColumn:
    """A trainer (registered as `predict_negated_column`) that predicts `-column`."""
    return PredictColumn(column, sign=-1.0)


def predict_column_spec(name: str, features: str, column: str, *, negated: bool = False) -> ModelSpec:
    """A `ModelSpec` for a fake trainer — needs the `fake_trainers` fixture."""
    return ModelSpec(
        name=name,
        features=features,
        trainer="predict_negated_column" if negated else "predict_column",
        params={"column": column},
    )
