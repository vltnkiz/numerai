"""RunConfig: the single dataclass that parameterizes one pipeline invocation.

Composed of nested sub-configs per docs/adr/0001-zemir-pipeline-architecture.md
so the same shape covers linear regression today and era-boosted XGBoost once
[Second model integration](https://github.com/vltnkiz/numerai/issues/12) plugs
it in behind `EraBoostConfig`. Built once by the caller and threaded down into
every stage by zemir/pipeline.py — not owned by a `Model` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class FeatureConfig:
    """Which columns a run trains/predicts on and which ones neutralize against.

    `feature_set` selects a key from features.json's `feature_sets` (passed to
    zemir/download.py's `download(feature_set=...)`); `neutralizers` defaults
    to the same columns the model was trained on when left unset — pass an
    explicit list to neutralize against a different (e.g. smaller) set.
    """

    feature_set: str = "small"
    neutralizers: list[str] | None = None


@dataclass
class EraBoostConfig:
    """Era-boosted XGBoost's hyperparameters, per era_boost_train_with_history in
    era-boosting/code/era_boosting.ipynb.

    Passed to `zemir.models.era_boost.EraBoostModel` at construction (per
    docs/adr/0001-zemir-pipeline-architecture.md: config isn't owned by the
    `Model` instance, but the model still needs it to configure itself).
    `RunConfig.era_boost` stays `None` for linear-regression runs.
    """

    trees_per_step: int = 50
    num_iters: int = 40
    snapshot_every: int = 5
    proportion: float = 0.5
    learning_rate: float = 0.01
    max_depth: int = 5
    colsample_bytree: float = 0.1


@dataclass
class NeutralizationConfig:
    proportion: float = 0.5


@dataclass
class RunConfig:
    data_version: str = "5.0"
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    features: FeatureConfig = field(default_factory=FeatureConfig)
    era_boost: EraBoostConfig | None = None
    neutralization: NeutralizationConfig = field(default_factory=NeutralizationConfig)
    min_validation_mean_corr: float = 0.0
