"""Named pipeline configurations.

One flat, frozen `PipelineConfig` describes everything about a run except which
models to fit and where to write. Named profiles (`LIVE`, `SMOKE`) are instances
of it; a sweep builds more with `dataclasses.replace(LIVE, ...)`.

Deliberately flat: no sub-configs, and no `run_id` — a config names a
*configuration*, not an instance of running one. The submission slot and the
validation gate live here as plain constants rather than config fields, because
`run_pipeline` neither submits nor gates; only scripts/run_pipeline.py does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from zemir.models import Trainer, train_linear, train_xgboost

XGBOOST_HYPERPARAMS: Mapping[str, object] = {
    "trees_per_step": 50,
    "num_iters": 40,
    "proportion": 0.5,
    "learning_rate": 0.01,
    "max_depth": 5,
    "colsample_bytree": 0.1,
}

# Read only by scripts/run_pipeline.py — the one entrypoint that submits.
SUBMISSION_MODEL_SLOT = "zemir_01"
MIN_VALIDATION_MEAN_CORR = 0.0

MODEL_NAMES = ["linear", "era_boost", "ensemble"]


@dataclass(frozen=True)
class PipelineConfig:
    data_version: str = "5.0"
    feature_set: str = "small"
    neutralization_proportion: float = 0.5
    max_eras: int | None = None  # None = every era
    xgboost: Mapping[str, object] = field(
        default_factory=lambda: dict(XGBOOST_HYPERPARAMS)
    )


LIVE = PipelineConfig()

# Exercises the identical code path in seconds instead of an hour. Scores from a
# SMOKE run are meaningless and must never be compared against a LIVE-scale one.
SMOKE = replace(
    LIVE,
    max_eras=40,
    xgboost={**XGBOOST_HYPERPARAMS, "num_iters": 2},
)


def build_trainers(model: str, config: PipelineConfig) -> dict[str, Trainer]:
    """Trainers for one model choice, using `config`'s hyperparameters."""

    def linear(X, y, era):
        return train_linear(X, y)

    def era_boost(X, y, era):
        return train_xgboost(X, y, era, **config.xgboost)

    by_name: dict[str, dict[str, Trainer]] = {
        "linear": {"linear": linear},
        "era_boost": {"era_boost": era_boost},
        "ensemble": {"linear": linear, "era_boost": era_boost},
    }
    if model not in by_name:
        raise ValueError(f"unknown model {model!r} (have: {MODEL_NAMES})")
    return by_name[model]


@dataclass(frozen=True)
class ScoringConfig:
    """One row of the comparison table: what to score, downstream of a fit.

    Separate from `PipelineConfig` because it names a different stage. A
    `PipelineConfig` describes a *fit* — an hour of training, recorded once in
    the harness cache's `fit_config.json` and immutable thereafter. A
    `ScoringConfig` describes what is done to those cached predictions, costs
    seconds, and is swept many times against one fit.

    Flat, like `PipelineConfig`: `proportion = 0.0` *is* "no neutralization"
    (the correction term is exactly zero), so there is no separate on/off field.
    """

    name: str
    models: tuple[str, ...]
    neutralization_proportion: float = 0.0


def scoring_sweep(
    model_sets: tuple[tuple[str, ...], ...] = (
        ("linear",),
        ("era_boost",),
        ("linear", "era_boost"),
    ),
    proportions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[ScoringConfig]:
    """Every combination of models and neutralization proportion, named `<models>_p<prop>`."""
    return [
        ScoringConfig(f"{'+'.join(models)}_p{proportion:g}", models, proportion)
        for models in model_sets
        for proportion in proportions
    ]


# Today's live submission — linear only, neutralized at 0.5. The row every later
# comparison on this map is measured against; a member of `scoring_sweep()`, not
# a duplicate of one.
PRODUCTION_BASELINE = "linear_p0.5"
