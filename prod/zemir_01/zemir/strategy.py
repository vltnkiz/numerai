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

from collections.abc import Mapping
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

    `features` is a real field even though every shipped spec names
    `"medium"` for now — per-model feature widths are #80's machinery, not
    this ticket's; the field exists now so nothing downstream has to add it
    later.

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


def build_trainer(spec: ModelSpec) -> Trainer:
    """A `zemir.models.Trainer` that fits `spec.trainer` with `spec.params` baked in."""
    fit = TRAINERS[spec.trainer]

    def trainer(X: pd.DataFrame, y: pd.Series, era: pd.Series) -> object:
        return fit(X, y, era, **spec.params)

    return trainer


def build_trainers(strategy: Strategy) -> dict[str, Trainer]:
    """One `Trainer` per `ModelSpec` in `strategy`, keyed by its own name.

    Same shape as `zemir.config.build_trainers`'s return value, so it drops
    straight into `zemir.pipeline.fit_models`/`predict_each` unchanged — two
    `ModelSpec`s naming the same `trainer` (e.g. two `era_boost`s at
    different depths) collide on trainer name today; they don't collide here,
    since each keys off its own `ModelSpec.name` instead.
    """
    return {spec.name: build_trainer(spec) for spec in strategy.models}
