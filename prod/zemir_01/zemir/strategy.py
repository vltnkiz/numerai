"""The Strategy spec: which models, blended how.

`zemir.config.PipelineConfig` stays *which data, which window*; a `Strategy`
is *which models, blended how* — the parameters a research idea needs to
vary per-model (feature set, model params, ensemble weight, neutralization)
live on `ModelSpec`/`BlendSpec` instead of the three tables `config.py` keys
by `MODEL_NAMES` today.

Neutralization is the one parameter that can be applied at two stages — to each
model's predictions before they are blended, and to the blend after — so it is
its own small type (`Neutralization`) that both specs hold, rather than a pair
of fields each spec spells its own way (issue #88).

Deliberately its own module, not part of `config.py`: `config.py` is named
*instances* with long measurement rationales attached, while these are
*types*. Importing this module must never drag in `neutralizer_ranking.py`'s
809 lines, so tests can import it on its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from zemir.models import FittedModel, Trainer, train_linear, train_ridge, train_sgd, train_xgboost


def _fit_ols(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> FittedModel:
    return train_linear(X, y, **params)


def _fit_ridge(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> FittedModel:
    return train_ridge(X, y, **params)


def _fit_sgd(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> FittedModel:
    return train_sgd(X, y, era, **params)


def _fit_xgboost(X: pd.DataFrame, y: pd.Series, era: pd.Series, **params: object) -> FittedModel:
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
class Neutralization:
    """Remove `proportion` of a prediction's linear component explained by `features`.

    `features` is required and explicit: either a feature-set name, resolved the
    way `ModelSpec.features` is, or a tuple of column names. There is no default
    set — "neutralize against everything" used to mean the run's loaded columns
    in the live run and `PipelineConfig.feature_set` in the harness, and the two
    disagreed for any strategy whose models were narrower than the run's
    feature set (issue #88). A stage that neutralizes nothing holds `None`
    instead of a `Neutralization`, so "no neutralization" and "neutralize
    against a set I forgot to name" cannot be confused.

    The proportion is applied per era. `proportion=0.0` is arithmetically a
    no-op, but write `None` for that — a sweep row for it should cost no solve.
    """

    proportion: float
    features: str | tuple[str, ...]


@dataclass(frozen=True)
class ModelSpec:
    """One model: its own feature width, trainer, hyperparameters and neutralization.

    `trainer` names an entry in `TRAINERS` rather than holding the callable
    itself — a callable would break `json.dumps(asdict(...))` (see
    `zemir.pipeline._write_run_config`), would not compare equal by value the
    way a plain string does, and would forfeit the "a Strategy is plain,
    serializable data" property this module exists for.

    `features` names a feature set (`"small"`/`"medium"`/`"all"`), and is what
    the model is actually fitted and predicted on — `zemir.fitting` hands each
    spec its own slice of the union of every spec's columns (#80).
    `PipelineConfig.feature_set` is a different thing: the universe the harness
    measures feature exposure against, not what any model trains on and not
    what anything is neutralized against (a `Neutralization` names its own).

    `neutralization` is applied to this model's predictions *before* the blend
    (`zemir.blend`); `None` applies nothing. Unlike the three fields above it
    is not a property of the fit — it is applied to cached predictions in
    seconds, which is why a harness cache can be rescored under a different one
    (`zemir.harness.score_configs`) but not under a different `params`.

    There is no feature-transform field, and that is a decision, not an
    omission (issue #90): the transform is fixed by the trainer, not chosen per
    model. `sgd` scales its int8 `{0..4}` features to `[-1, 1]` because SGD
    needs it, and `FittedModel.transform` applies that one rule at fit and at
    predict. On the other trainers a chosen transform would do nothing
    (`xgboost`, `ols`) or restate `alpha` (`ridge`: scaled `Ridge(a)` is exactly
    unscaled `Ridge(4a)`), so a field would be honoured by one trainer of four
    — a table two-thirds full of `None`. If feature engineering ever needs to
    vary, look upstream of the model (the feature set, the data layer) first.
    """

    name: str
    features: str
    trainer: str
    params: Mapping[str, object] = field(default_factory=dict)
    neutralization: Neutralization | None = None


@dataclass(frozen=True)
class BlendSpec:
    """How a Strategy's fitted models combine into one prediction.

    `neutralization` is applied to the blended prediction; `None` applies
    nothing.
    """

    weights: Mapping[str, float] | None = None
    neutralization: Neutralization | None = None


@dataclass(frozen=True)
class Strategy:
    """Which models, blended how."""

    models: tuple[ModelSpec, ...]
    blend: BlendSpec

    @property
    def neutralizations(self) -> tuple[Neutralization, ...]:
        """Every neutralization this strategy applies — each model's, then the blend's."""
        found = [spec.neutralization for spec in self.models] + [self.blend.neutralization]
        return tuple(n for n in found if n is not None)

    @property
    def feature_set_names(self) -> tuple[str, ...]:
        """Every distinct feature set named, by a model or a neutralization, in first-seen order."""
        named = [spec.features for spec in self.models]
        named += [n.features for n in self.neutralizations if isinstance(n.features, str)]
        return tuple(dict.fromkeys(named))


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


def neutralizer_columns(neutralization: Neutralization, feature_sets: FeatureSets) -> tuple[str, ...]:
    """The columns `neutralization` projects onto: a named set's, or its own tuple as written."""
    features = neutralization.features
    if not isinstance(features, str):
        return tuple(features)
    try:
        return tuple(feature_sets[features])
    except KeyError:
        raise KeyError(
            f"a neutralization names feature set {features!r}, which was not "
            f"resolved (have: {sorted(feature_sets)})"
        ) from None


def required_columns(strategy: Strategy, feature_sets: FeatureSets) -> list[str]:
    """What a live run must load: the training union, plus every column a neutralization names.

    `run_pipeline` reads the neutralizer columns off the live frame, and a
    strategy whose models are narrower than what it neutralizes against would
    otherwise fail there rather than at load.
    """
    neutralizers = (
        c for n in strategy.neutralizations for c in neutralizer_columns(n, feature_sets)
    )
    return list(dict.fromkeys((*union_columns(strategy, feature_sets), *neutralizers)))


def build_trainer(spec: ModelSpec) -> Trainer:
    """A `zemir.models.Trainer` that fits `spec.trainer` with `spec.params` baked in."""
    fit = TRAINERS[spec.trainer]

    def trainer(X: pd.DataFrame, y: pd.Series, era: pd.Series) -> FittedModel:
        return fit(X, y, era, **spec.params)

    return trainer


def strategy_from_record(record: Mapping[str, object]) -> Strategy:
    """A `Strategy` rebuilt from `dataclasses.asdict`'s JSON round trip (`fit_config.json`'s `strategy`).

    JSON turns tuples into lists and knows nothing of these dataclasses, so the
    harness, which learns what a cache was fitted with only from that file,
    needs this to compare a swept strategy against it (issue #88). Raises
    `KeyError` on a record written before `Neutralization` existed (its blend
    carries `neutralize`/`proportion` instead of `neutralization`); callers
    that must accept those treat it as "no record".
    """

    def neutralization(raw: Mapping[str, object] | None) -> Neutralization | None:
        if raw is None:
            return None
        features = raw["features"]
        return Neutralization(
            proportion=raw["proportion"],
            features=features if isinstance(features, str) else tuple(features),
        )

    blend = record["blend"]
    weights = blend["weights"]
    return Strategy(
        models=tuple(
            ModelSpec(
                name=m["name"],
                features=m["features"],
                trainer=m["trainer"],
                params=dict(m["params"]),
                neutralization=neutralization(m["neutralization"]),
            )
            for m in record["models"]
        ),
        blend=BlendSpec(
            weights=dict(weights) if weights is not None else None,
            neutralization=neutralization(blend["neutralization"]),
        ),
    )
