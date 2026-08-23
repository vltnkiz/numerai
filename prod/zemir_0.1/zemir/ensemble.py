"""Ensembling stage: combine multiple models' pure predictions into one series.

Per docs/adr/0001-zemir-pipeline-architecture.md, `Model.predict()` is kept
pure/stateless specifically so combining predictions needs no `Model`
protocol changes — just this pipeline-level rank average.
"""

from __future__ import annotations

import pandas as pd


def combine_predictions(predictions: dict[str, pd.Series], era: pd.Series) -> pd.Series:
    """Equal-weight average of each named model's predictions, ranked per era first.

    Ranking (percentile, within each era) before averaging makes the result
    robust to different prediction scales/distributions across model types —
    a plain average of raw scores would let whichever model happens to
    produce a wider spread dominate.
    """
    ranked = {name: preds.groupby(era).rank(pct=True) for name, preds in predictions.items()}
    combined = sum(ranked.values()) / len(ranked)
    return combined.rename("prediction")
