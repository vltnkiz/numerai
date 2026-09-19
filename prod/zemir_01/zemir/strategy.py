"""The Strategy spec: which models, blended how.

`zemir.config.PipelineConfig` stays *which data, which window*; a `Strategy`
is *which models, blended how* — the six parameters a research idea needs to
vary per-model (feature set, model params, ensemble weight, neutralization)
live on `ModelSpec`/`BlendSpec` instead of the three tables `config.py` keys
by `MODEL_NAMES` today.

Deliberately its own module, not part of `config.py`: `config.py` is named
*instances* with long measurement rationales attached, while these are
*types*. Importing this module must never drag in `neutralizer_ranking.py`'s
809 lines, so tests can import it on its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from zemir.models import Trainer, train_linear, train_ridge, train_sgd, train_xgboost


def _fit_ols(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> object:
    return train_linear(X, y, **params)


def _fit_ridge(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> object:
    return train_ridge(X, y, **params)


def _fit_sgd(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> object:
    return train_sgd(X, y, era, **params)


def _fit_xgboost(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> object:
    return train_xgboost(X, y, era, **params)


# Every name a ModelSpec.trainer can resolve to. Each entry is shaped to
# zemir.models.Trainer's (features, target, era) contract regardless of what
# the underlying train_* function itself takes — train_linear and train_ridge
# accept no `era`, train_ridge's `alpha` is a required keyword, not a default
# — so that mismatch is absorbed once here rather than at every call site.
TRAINERS: Mapping[str, Trainer] = {
    "ols": _fit_ols,
    "ridge": _fit_ridge,
    "sgd": _fit_sgd,
    "xgboost": _fit_xgboost,
}


@dataclass(frozen=True)
class ModelSpec:
    """One model: its own feature width, trainer, hyperparameters and neutralizer subset.

    `trainer` names an entry in `TRAINERS` rather than holding the callable
    itself — a callable would break `json.dumps(asdict(...))` (see
    `zemir.pipeline._write_run_config`), would not compare equal by value the
    way a plain string does, and would forfeit the "a Strategy is plain,
    serializable data" property this module exists for.

    `features` names a feature set (`"small"`/`"medium"`/`"all"`), and is what
    the model is actually fitted and predicted on — `zemir.fitting` hands each
    spec its own slice of the union of every spec's columns (#80).
    `PipelineConfig.feature_set` is a different thing: the run's default
    universe for scoring and neutralization, not what any model trains on.

    `engineering` (e.g. `train_sgd`'s fixed feature scaling) is deliberately
    absent: no code would read it yet (see #74's Not-yet-specified), and a
    field nothing reads is the same defect as a table two-thirds full of None.
    """

    name: str
    features: str
    trainer: str
    params: Mapping[str, object] = field(default_factory=dict)
    neutralize: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BlendSpec:
    """How a Strategy's fitted models combine into one prediction.

    Flat, like `zemir.config.ScoringConfig`: `proportion=0.0` *is* "no
    neutralization", so there is no separate on/off field.
    """

    weights: Mapping[str, float] | None = None
    neutralize: tuple[str, ...] | None = None
    proportion: float = 0.0


@dataclass(frozen=True)
class Strategy:
    """Which models, blended how."""

    models: tuple[ModelSpec, ...]
    blend: BlendSpec

    @property
    def feature_set_names(self) -> tuple[str, ...]:
        """Every distinct feature set a model names, in first-seen order."""
        return tuple(dict.fromkeys(spec.features for spec in self.models))


# Feature-set name -> its columns, as `zemir.data.resolve_feature_sets` builds it.
FeatureSets = Mapping[str, Sequence[str]]


def model_columns(spec: ModelSpec, feature_sets: FeatureSets) -> tuple[str, ...]:
    """The columns `spec` is fitted and predicted on."""
    try:
        return tuple(feature_sets[spec.features])
    except KeyError:
        raise KeyError(
            f"model {spec.name!r} names feature set {spec.features!r}, which was not "
            f"resolved (have: {sorted(feature_sets)})"
        ) from None


def union_columns(strategy: Strategy, feature_sets: FeatureSets) -> tuple[str, ...]:
    """Every column any model trains on, once each, in first-seen order.

    The order is load-bearing, not cosmetic: a model whose columns are the
    whole union is handed the union frame itself (see `zemir.fitting`), and
    tree stages sample columns by position — so a strategy whose models all
    name one feature set must see that set's own column order, unchanged.
    """
    return tuple(
        dict.fromkeys(c for spec in strategy.models for c in model_columns(spec, feature_sets))
    )


def required_columns(strategy: Strategy, feature_sets: FeatureSets) -> list[str]:
    """What a live run must load: the training union, plus blend neutralizers outside it.

    `run_pipeline` reads the neutralizer columns off the live frame, and a
    strategy whose models are narrower than its neutralizer set would
    otherwise fail there rather than at load.
    """
    return list(
        dict.fromkeys((*union_columns(strategy, feature_sets), *(strategy.blend.neutralize or ())))
    )


def build_trainer(spec: ModelSpec) -> Trainer:
    """A `zemir.models.Trainer` that fits `spec.trainer` with `spec.params` baked in."""
    fit = TRAINERS[spec.trainer]

    def trainer(X: pd.DataFrame, y: pd.Series, era: pd.Series) -> object:
        return fit(X, y, era, **spec.params)

    return trainer
