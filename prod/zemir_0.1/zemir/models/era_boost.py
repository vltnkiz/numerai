"""Era-boosted XGBoost Model implementation.

Ports era_boost_train_with_history from era-boosting/code/era_boosting.ipynb
onto the Model protocol (zemir/models/base.py): fit an XGBRegressor, then
repeatedly re-fit it on only the worst-scoring `proportion` of eras (by
per-era Spearman corr against `y`), growing the booster incrementally via
`xgb_model=` rather than retraining from scratch each iteration.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from zemir.config import EraBoostConfig
from zemir.models.base import FitResult


class EraBoostModel:
    def __init__(self, config: EraBoostConfig | None = None) -> None:
        self._config = config or EraBoostConfig()
        self._model: XGBRegressor | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        *,
        checkpoint_path: Path | None = None,
    ) -> FitResult:
        model, history = _era_boost_train_with_history(
            X, y, era, self._config, checkpoint_path=checkpoint_path
        )
        self._model = model
        return FitResult(model=model, history=history)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


def _era_boost_train_with_history(
    X: pd.DataFrame,
    y: pd.Series,
    era: pd.Series,
    config: EraBoostConfig,
    *,
    checkpoint_path: Path | None,
) -> tuple[XGBRegressor, pd.DataFrame]:
    working = X.copy()
    working["target"] = y
    working["era"] = era

    start_iter = 0
    history: list[dict] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = joblib.load(checkpoint_path)
        model = checkpoint["model"]
        start_iter = checkpoint["iter"]
        history = checkpoint["history"]
    else:
        model = XGBRegressor(
            n_estimators=config.trees_per_step,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            colsample_bytree=config.colsample_bytree,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X, y)

    for i in range(start_iter, config.num_iters):
        working["pred"] = model.predict(X)
        era_scores = working.groupby("era").apply(
            lambda d: spearmanr(d["pred"], d["target"])[0]
        ).to_dict()

        if (i + 1) % config.snapshot_every == 0 or i == 0:
            for e, corr in era_scores.items():
                history.append({"iter": i + 1, "era": e, "corr": corr})

        worst_eras = sorted(era_scores, key=era_scores.get)[
            : int(len(era_scores) * config.proportion)
        ]
        worst_df = working[working["era"].isin(worst_eras)]

        # xgb_model=booster continues training and adds `n_estimators` NEW
        # trees on top of the existing booster, not a target total — so
        # n_estimators stays pinned at trees_per_step every iteration.
        booster = model.get_booster()
        model.n_estimators = config.trees_per_step
        model.fit(
            worst_df.drop(columns=["target", "era", "pred"]),
            worst_df["target"],
            xgb_model=booster,
        )

        if checkpoint_path is not None and (i + 1) % config.snapshot_every == 0:
            joblib.dump(
                {"model": model, "iter": i + 1, "history": history},
                checkpoint_path,
            )

    return model, pd.DataFrame(history)
