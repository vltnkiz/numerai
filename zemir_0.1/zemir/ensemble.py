"""Ensembling stage: combine multiple models' pure predictions into one series.

Per docs/adr/0001-zemir-pipeline-architecture.md, `Model.predict()` is kept
pure/stateless specifically so combining predictions needs no `Model`
protocol changes — just this pipeline-level weighted average.
"""

from __future__ import annotations

import pandas as pd


def combine_predictions(
    predictions: dict[str, pd.Series], weights: dict[str, float]
) -> pd.Series:
    """Weighted average of each named model's predictions.

    `weights` are normalized to sum to 1 first, so callers can pass either
    proportions (0.3/0.7) or arbitrary ratios (30/70) — same result either
    way. `predictions` and `weights` must name the same set of models.
    """
    if predictions.keys() != weights.keys():
        raise ValueError(
            f"predictions and weights must name the same models: "
            f"{sorted(predictions)} vs {sorted(weights)}"
        )

    total_weight = sum(weights.values())
    combined = sum(
        predictions[name] * (weight / total_weight) for name, weight in weights.items()
    )
    return combined.rename("prediction")
