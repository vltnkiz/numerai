"""Score the pipeline **as actually submitted**, over validation eras.

`run_pipeline` scores raw per-model predictions — the artifact nobody uploads.
This harness scores the uploaded one: combine → neutralize → rank-normalize,
on Numerai's *paid* metrics (`numerai_corr` and MMC), so "is A better than B"
is answerable with a number.

Split in two stages, because fitting is the only expensive part:

1. `fit_validation_predictions` — trains each model on the train eras exactly as
   production does (same trainers, same frames, via `zemir.pipeline`) and caches
   its raw validation predictions to parquet. ~1 hour with era boosting.
2. `score_configs` — sweeps combination rules, neutralizers and proportions over
   those cached predictions. Seconds, and repeatable without refitting.

Anything that changes a *model* (hyperparameters, feature set, training eras)
invalidates the cache and needs a new fit. Anything downstream of `.predict()`
is free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from zemir.config import PipelineConfig, ScoringConfig, build_trainers
from zemir.data import download, load_meta_model, load_validation_features
from zemir.pipeline import (
    RUNS_DIR,
    combine_predictions,
    fit_models,
    predict_each,
    scoring_frames,
)
from zemir.scoring import (
    era_feature_projection,
    era_max_feature_corr,
    era_mmc,
    era_numerai_corr,
    summarize_era_scores,
)

HARNESS_DIR = RUNS_DIR / "harness"
PREDICTIONS_FILENAME = "validation_predictions.parquet"


@dataclass
class FitResult:
    run_dir: Path
    predictions_path: Path
    predictions: pd.DataFrame


def fit_validation_predictions(
    config: PipelineConfig,
    *,
    run_id: str,
    model: str = "ensemble",
    harness_dir: Path = HARNESS_DIR,
) -> FitResult:
    """Fit production's models and cache their raw validation predictions.

    The expensive stage. Writes `validation_predictions.parquet` (index `id`;
    columns `era`, `target`, one per model) plus the `fit_config.json` that
    produced it — a scoring run stamps that config onto its results so a
    comparison is never read against the wrong fit.
    """
    run_dir = harness_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = download(config.data_version, config.feature_set)
    train_df, validation_df = scoring_frames(dataset, config)
    trainers = build_trainers(model, config)

    fitted = fit_models(trainers, train_df, dataset.feature_columns)
    predictions = predict_each(fitted, validation_df, dataset.feature_columns)

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
                "max_eras": config.max_eras,
                "xgboost": dict(config.xgboost),
                "models": sorted(trainers),
                "train_eras": len(train_df["era"].unique()),
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
    """
    cache = pd.read_parquet(run_dir / PREDICTIONS_FILENAME)
    fit = json.loads((run_dir / "fit_config.json").read_text())

    eras = sorted(cache["era"].unique(), key=int)
    features = load_validation_features(
        fit["data_version"], fit["feature_set"], eras=eras
    ).loc[cache.index]
    neutralizers = [c for c in features.columns if c != "era"]
    meta_model = load_meta_model(fit["data_version"])

    model_columns = [c for c in cache.columns if c not in ("era", "target")]
    model_projections = era_feature_projection(
        pd.concat([features, cache[model_columns]], axis=1), model_columns, neutralizers
    )

    blends = _blends(configs, cache)
    blend_projections = era_feature_projection(
        pd.concat([features, blends], axis=1), list(blends.columns), neutralizers
    )

    def _score(config: ScoringConfig) -> pd.Series:
        if config.neutralize_before_blend:
            neutralized = {
                name: cache[name] - config.neutralization_proportion * model_projections[name]
                for name in config.models
            }
            return combine_predictions(neutralized, cache["era"])
        blend_name = _blend_name(config.models)
        return blends[blend_name] - config.neutralization_proportion * blend_projections[blend_name]

    predictions = pd.DataFrame(
        {config.name: _score(config).astype("float32") for config in configs}
    )

    corr_by_era = era_numerai_corr(predictions, cache["target"], cache["era"])
    mmc_by_era = era_mmc(predictions, cache["target"], cache["era"], meta_model)
    exposure_by_era = era_max_feature_corr(
        predictions, features[neutralizers], cache["era"]
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


def _blend_name(models: tuple[str, ...]) -> str:
    return "+".join(models)


def _blends(configs: list[ScoringConfig], cache: pd.DataFrame) -> pd.DataFrame:
    """One column per *distinct* model set in the sweep, combined as production combines.

    Keyed by model set rather than by config, because the neutralization
    proportion is applied afterwards — every config sharing a model set shares
    one blend and one projection.
    """
    model_sets = dict.fromkeys(config.models for config in configs)
    missing = sorted(
        {name for models in model_sets for name in models} - set(cache.columns)
    )
    if missing:
        raise KeyError(f"model(s) {missing} are not in this fit's cache")

    return pd.DataFrame(
        {
            _blend_name(models): combine_predictions(
                {name: cache[name] for name in models}, cache["era"]
            )
            for models in model_sets
        }
    )
