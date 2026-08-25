from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# (features, target, era) -> a fitted object exposing .predict(features)
Trainer = Callable[[pd.DataFrame, pd.Series, pd.Series], object]


def train_linear(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    era: pd.Series,
    *,
    trees_per_step: int = 50,
    num_iters: int = 40,
    proportion: float = 0.5,
    learning_rate: float = 0.01,
    max_depth: int = 5,
    colsample_bytree: float = 0.1,
) -> XGBRegressor:
    """Fit, then repeatedly re-fit on the worst-scoring `proportion` of eras."""
    model = XGBRegressor(
        n_estimators=trees_per_step,
        learning_rate=learning_rate,
        max_depth=max_depth,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X, y)

    working = X.copy()
    working["target"] = y
    working["era"] = era

    for _ in range(num_iters):
        working["pred"] = model.predict(X)
        era_scores = (
            working.groupby("era")
            .apply(lambda d: spearmanr(d["pred"], d["target"])[0])
            .to_dict()
        )
        worst_eras = sorted(era_scores, key=era_scores.get)[
            : int(len(era_scores) * proportion)
        ]
        worst_df = working[working["era"].isin(worst_eras)]

        # xgb_model= adds `n_estimators` NEW trees on top of the existing
        # booster, not a target total, so it stays pinned at trees_per_step.
        booster = model.get_booster()
        model.n_estimators = trees_per_step
        model.fit(
            worst_df.drop(columns=["target", "era", "pred"]),
            worst_df["target"],
            xgb_model=booster,
        )

    return model
