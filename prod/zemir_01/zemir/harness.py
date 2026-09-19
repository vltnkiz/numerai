"""Score the pipeline **as actually submitted**, over validation eras.

`run_pipeline` scores raw per-model predictions — the artifact nobody uploads.
This harness scores the uploaded one: combine → neutralize → rank-normalize,
on Numerai's *paid* metrics (`numerai_corr` and MMC), so "is A better than B"
is answerable with a number.

Split in two stages, because fitting is the only expensive part:

1. `fit_validation_predictions` — trains each model on the train eras exactly as
   production does (same trainers, same frames, via `zemir.pipeline`) and caches
   its raw validation predictions to parquet.
2. `score_configs` — sweeps combination rules, neutralizers and proportions over
   those cached predictions. Seconds, and repeatable without refitting.

Anything that changes a *model* (hyperparameters, feature set, training eras)
invalidates the cache and needs a new fit. Anything downstream of `.predict()`
is free.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from zemir.config import PipelineConfig, ScoringConfig
from zemir.data import load_meta_model, load_validation_features, scoring_window
from zemir.fitting import fit_strategy
from zemir.pipeline import RUNS_DIR, combine_predictions, predict_each
from zemir.scoring import (
    era_feature_corr,
    era_feature_projection,
    era_max_feature_corr,
    era_mmc,
    era_numerai_corr,
    summarize_era_scores,
)
from zemir.strategy import FeatureSets, Strategy, union_columns

HARNESS_DIR = RUNS_DIR / "harness"
PREDICTIONS_FILENAME = "validation_predictions.parquet"


@dataclass
class FitResult:
    run_dir: Path
    predictions_path: Path
    predictions: pd.DataFrame


def fit_validation_predictions(
    config: PipelineConfig,
    strategy: Strategy,
    *,
    run_id: str,
    feature_sets: FeatureSets,
    load_train: Callable[[], pd.DataFrame],
    load_validation: Callable[[], pd.DataFrame],
    harness_dir: Path = HARNESS_DIR,
) -> FitResult:
    """Fit `strategy`'s models and cache their raw validation predictions.

    The expensive stage. Writes `validation_predictions.parquet` (index `id`;
    columns `era`, `target`, one per model) plus the `fit_config.json` that
    produced it — a scoring run stamps that config onto its results so a
    comparison is never read against the wrong fit.

    `strategy` names its own trainers via `ModelSpec.trainer`, resolved
    through `zemir.strategy.TRAINERS` — trying a regularized linear stage
    (issue #38) or a chunked one (issue #50) is a caller building its own
    `Strategy` with `linear`'s spec swapped for a different trainer name,
    rather than overriding a `build_trainers` callable.

    Takes `feature_sets` and a `load_train`/`load_validation` pair of
    zero-argument callables rather than a `Dataset` (contrast
    `zemir.pipeline.run_pipeline`, which does take one): the caller builds
    each loader around its own I/O (a real `load_split` call bound to
    `union_columns(strategy, feature_sets)` — already narrowed for issue #45's
    dropped columns before either loader is ever called — in production; a
    synthetic fixture in a test), but *when* each is called stays right here,
    one at a time — train first, fitted and freed inside `fit_strategy`
    (`zemir.fitting`, which owns that discipline), then validation — so a
    `Dataset` argument that forced both splits resident up front could never
    silently undo issue #47/#51's memory discipline.
    """
    run_dir = harness_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fit = fit_strategy(strategy, load_train, feature_sets=feature_sets, max_eras=config.max_eras)

    validation_df = scoring_window(load_validation(), config.max_eras)

    predictions = predict_each(fit.models, validation_df)

    frame = validation_df[["era", "target"]].copy()
    for name, preds in predictions.items():
        frame[name] = preds.astype("float32")

    predictions_path = run_dir / PREDICTIONS_FILENAME
    frame.to_parquet(predictions_path)
    (run_dir / "fit_config.json").write_text(
        json.dumps(
            {
                "data_version": config.data_version,
                "feature_set": config.feature_set,
                "feature_count": len(union_columns(strategy, feature_sets)),
                "max_eras": config.max_eras,
                "strategy": asdict(strategy),
                "train_eras": fit.train_eras,
                "validation_eras": len(frame["era"].unique()),
                "validation_rows": len(frame),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return FitResult(run_dir=run_dir, predictions_path=predictions_path, predictions=frame)


@dataclass
class ScoreResult:
    run_dir: Path
    summary: pd.DataFrame
    era_corr: pd.DataFrame
    era_mmc: pd.DataFrame
    era_max_feature_corr: pd.DataFrame


def _load_cache_and_features(run_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, list[str]]:
    """The cache, its own fit config, validation features, and the full neutralizer set.

    Shared by `score_configs` and `rank_feature_exposure` so both read the
    dataset the cache was actually fit on, never a guessed one.
    """
    cache = pd.read_parquet(run_dir / PREDICTIONS_FILENAME)
    fit = json.loads((run_dir / "fit_config.json").read_text())
    eras = sorted(cache["era"].unique(), key=int)
    features = load_validation_features(
        fit["data_version"], fit["feature_set"], eras=eras
    ).loc[cache.index]
    full_neutralizers = [c for c in features.columns if c != "era"]
    return cache, fit, features, full_neutralizers


def rank_feature_exposure(
    run_dir: Path,
    *,
    models: tuple[str, ...] = ("linear", "era_boost"),
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Mean absolute exposure of every feature to one blend, across every validation era.

    Answers "exposure to what" for issue #35: ranks the pre-neutralization
    blend — the artifact neutralization actually corrects — so a deliberate
    neutralizer subset can target the features it is measurably most exposed
    to, rather than a guessed or merely-published group. Descending order:
    `.index[:k]` is the top-k most-exposed feature set.
    """
    cache, _, features, full_neutralizers = _load_cache_and_features(run_dir)
    blend = combine_predictions({name: cache[name] for name in models}, cache["era"], weights)
    exposure = era_feature_corr(blend, features[full_neutralizers], cache["era"])
    return exposure.mean().sort_values(ascending=False)


def score_configs(
    configs: list[ScoringConfig],
    *,
    run_dir: Path,
) -> ScoreResult:
    """Score every configuration against one cached fit, on Numerai's paid metrics.

    The cheap stage: applies the rest of the submitted transform — combine, then
    neutralize per era exactly as production neutralizes the single live era —
    and scores the result. `rank_normalize` is deliberately *not* applied: it is
    a strictly monotone transform, and CORR, MMC and the exposure measure here
    all rank internally, so it is provably free (issue #26).

    Reads which dataset produced the cache from its own `fit_config.json`, so a
    sweep can never be scored against the wrong features or the wrong meta model.

    Each config may name its own neutralizer subset (`ScoringConfig.neutralizers`,
    issue #35); `None` falls back to the full feature set, matching production.
    The `max_feature_corr` diagnostic always measures exposure to the *full*
    set regardless of what a config neutralized against — it is the total risk
    being reported on, not a number a subset gets to grade itself on.
    """
    cache, fit, features, full_neutralizers = _load_cache_and_features(run_dir)
    meta_model = load_meta_model(fit["data_version"])

    blends = _blends(configs, cache)

    def _neutralizers(config: ScoringConfig) -> tuple[str, ...]:
        return tuple(config.neutralizers) if config.neutralizers is not None else tuple(full_neutralizers)

    # Grouped by neutralizer set, not computed one column at a time: the sweep's
    # common case (every config sharing the full feature set) batches every
    # blend column into one era_feature_projection call per neutralizer group,
    # so the per-era lstsq factorization is solved once and reused across
    # columns — computed per column instead, this was ~3x slower measured on
    # score_harness.py's 15-config sweep (issue #35).
    def _grouped_projections(
        source: pd.DataFrame, groups: dict[tuple[str, ...], set[str]]
    ) -> dict[tuple[str, tuple[str, ...]], pd.Series]:
        result: dict[tuple[str, tuple[str, ...]], pd.Series] = {}
        for neutralizers, columns in groups.items():
            ordered = sorted(columns)
            projected = era_feature_projection(
                pd.concat([features, source[ordered]], axis=1), ordered, list(neutralizers)
            )
            for column in ordered:
                result[(column, neutralizers)] = projected[column]
        return result

    model_groups: dict[tuple[str, ...], set[str]] = {}
    blend_groups: dict[tuple[str, ...], set[str]] = {}
    for config in configs:
        neutralizers = _neutralizers(config)
        if config.neutralize_before_blend:
            model_groups.setdefault(neutralizers, set()).update(config.models)
        else:
            blend_groups.setdefault(neutralizers, set()).add(_blend_name(config.models, config.weights))

    model_projections = _grouped_projections(cache, model_groups)
    blend_projections = _grouped_projections(blends, blend_groups)

    def _score(config: ScoringConfig) -> pd.Series:
        neutralizers = _neutralizers(config)
        if config.neutralize_before_blend:
            neutralized = {
                name: cache[name]
                - config.neutralization_proportion * model_projections[(name, neutralizers)]
                for name in config.models
            }
            return combine_predictions(neutralized, cache["era"], _weights_map(config))
        blend_name = _blend_name(config.models, config.weights)
        return (
            blends[blend_name]
            - config.neutralization_proportion * blend_projections[(blend_name, neutralizers)]
        )

    predictions = pd.DataFrame(
        {config.name: _score(config).astype("float32") for config in configs}
    )

    corr_by_era = era_numerai_corr(predictions, cache["target"], cache["era"])
    mmc_by_era = era_mmc(predictions, cache["target"], cache["era"], meta_model)
    exposure_by_era = era_max_feature_corr(
        predictions, features[full_neutralizers], cache["era"]
    )
    summary = summarize_era_scores(corr_by_era, mmc_by_era, exposure_by_era)

    summary.to_csv(run_dir / "scores.csv")
    corr_by_era.to_csv(run_dir / "era_corr.csv")
    mmc_by_era.to_csv(run_dir / "era_mmc.csv")
    exposure_by_era.to_csv(run_dir / "era_max_feature_corr.csv")
    (run_dir / "scoring_config.json").write_text(
        json.dumps([asdict(config) for config in configs], indent=2)
    )
    return ScoreResult(
        run_dir=run_dir,
        summary=summary,
        era_corr=corr_by_era,
        era_mmc=mmc_by_era,
        era_max_feature_corr=exposure_by_era,
    )


def _weights_map(config: ScoringConfig) -> dict[str, float] | None:
    if config.weights is None:
        return None
    return dict(zip(config.models, config.weights))


def _blend_name(models: tuple[str, ...], weights: tuple[float, ...] | None = None) -> str:
    if weights is None:
        return "+".join(models)
    return "+".join(f"{name}{w:g}" for name, w in zip(models, weights))


def _blends(configs: list[ScoringConfig], cache: pd.DataFrame) -> pd.DataFrame:
    """One column per *distinct* (model set, weights) pair in the sweep.

    Keyed this way rather than by config, because the neutralization
    proportion is applied afterwards — every config sharing a model set and
    weighting shares one blend and one projection.
    """
    blend_keys = dict.fromkeys((config.models, config.weights) for config in configs)
    missing = sorted(
        {name for models, _ in blend_keys for name in models} - set(cache.columns)
    )
    if missing:
        raise KeyError(f"model(s) {missing} are not in this fit's cache")

    return pd.DataFrame(
        {
            _blend_name(models, weights): combine_predictions(
                {name: cache[name] for name in models},
                cache["era"],
                dict(zip(models, weights)) if weights is not None else None,
            )
            for models, weights in blend_keys
        }
    )
