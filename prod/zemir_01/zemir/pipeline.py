from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI

from zemir.config import PipelineConfig
from zemir.data import download
from zemir.models import Trainer
from zemir.scoring import (
    ValidationScore,
    neutralize_predictions,
    rank_normalize,
    score_validation,
    validate_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "prod" / "zemir_01" / "runs"


class ValidationScoreBelowThreshold(RuntimeError):
    pass


@dataclass
class SubmissionResult:
    model_name: str
    model_id: str
    submission_id: str


@dataclass
class PipelineResult:
    run_id: str
    run_dir: Path
    validation_scores: dict[str, ValidationScore]
    combined_validation_score: ValidationScore
    live_predictions: pd.Series
    live_predictions_neutralized: pd.Series


def _combine_predictions(predictions: dict[str, pd.Series], era: pd.Series) -> pd.Series:
    """Equal-weight average of each model's predictions, ranked per era first."""
    ranked = {name: preds.groupby(era).rank(pct=True) for name, preds in predictions.items()}
    combined = sum(ranked.values()) / len(ranked)
    return combined.rename("prediction")


def _load_numerai_models(env: dict[str, str] | None = None) -> dict[str, str]:
    env = env if env is not None else os.environ
    raw = env.get("NUMERAI_MODELS", "")
    if not raw.strip():
        raise ValueError("NUMERAI_MODELS is empty — add a model slot first")

    models: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, model_id = pair.partition("=")
        if not name or not model_id:
            raise ValueError(f"malformed NUMERAI_MODELS entry: {pair!r}")
        models[name] = model_id
    return models


def submit_predictions(
    predictions: pd.Series,
    *,
    model_slot: str,
    run_dir: Path,
    napi: NumerAPI | None = None,
) -> SubmissionResult:
    """Upload predictions to Numerai. THE ONLY FUNCTION IN THIS PACKAGE THAT SUBMITS.

    Deliberately not called by `run_pipeline`: an experiment that imports the
    pipeline cannot fire a live submission by forgetting a flag. Only
    scripts/run_pipeline.py — the entrypoint the scheduled workflow invokes —
    calls this.
    """
    load_dotenv()
    napi = napi or NumerAPI(
        public_id=os.environ["NUMERAI_PUBLIC_ID"],
        secret_key=os.environ["NUMERAI_SECRET_KEY"],
    )
    models = _load_numerai_models()
    if model_slot not in models:
        raise ValueError(
            f"model slot {model_slot!r} not found in NUMERAI_MODELS (have: {sorted(models)})"
        )

    frame = predictions.rename("prediction").reset_index()
    frame = frame.rename(columns={frame.columns[0]: "id"})[["id", "prediction"]]
    model_id = models[model_slot]
    submission_id = napi.upload_predictions(df=frame, model_id=model_id)
    submission = SubmissionResult(
        model_name=model_slot, model_id=model_id, submission_id=submission_id
    )
    (run_dir / "submission.json").write_text(
        json.dumps(
            {
                "model_name": submission.model_name,
                "model_id": submission.model_id,
                "submission_id": submission.submission_id,
            },
            indent=2,
        )
    )
    return submission


def run_pipeline(
    config: PipelineConfig,
    *,
    run_id: str,
    trainers: dict[str, Trainer],
    neutralizers: list[str] | None = None,
    runs_dir: Path = RUNS_DIR,
) -> PipelineResult:
    """Train, score, predict, neutralize and write artifacts.

    Never submits and never gates: submission is a separate step
    (`submit_predictions`), and the validation-score gate that guards it lives
    with it in scripts/run_pipeline.py.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = download(config.data_version, config.feature_set)
    feature_columns = dataset.feature_columns
    neutralizers = neutralizers or feature_columns

    train_df = dataset.train.dropna(subset=["target"])
    validation_df = dataset.validation.dropna(subset=["target"])
    if config.max_eras is not None:
        train_df = _last_eras(train_df, config.max_eras)
        validation_df = _last_eras(validation_df, config.max_eras)

    _write_run_config(run_dir, config, trainers=trainers, neutralizers=neutralizers)

    fitted_models = {
        name: trainer(train_df[feature_columns], train_df["target"], train_df["era"])
        for name, trainer in trainers.items()
    }

    validation_predictions = {
        name: pd.Series(
            model.predict(validation_df[feature_columns]),
            index=validation_df.index,
            name="prediction",
        )
        for name, model in fitted_models.items()
    }
    validation_scores = {
        name: score_validation(
            validation_df[["era", "target"]].assign(prediction=preds),
        )
        for name, preds in validation_predictions.items()
    }
    for name, score in validation_scores.items():
        _write_score(run_dir, score, prefix=f"{name}_")

    combined_validation_predictions = (
        next(iter(validation_predictions.values()))
        if len(validation_predictions) == 1
        else _combine_predictions(validation_predictions, validation_df["era"])
    )
    combined_validation_score = score_validation(
        validation_df[["era", "target"]].assign(prediction=combined_validation_predictions)
    )
    _write_score(run_dir, combined_validation_score, prefix="")

    live_predictions_by_model = {
        name: pd.Series(
            model.predict(dataset.live[feature_columns]),
            index=dataset.live.index,
            name="prediction",
        )
        for name, model in fitted_models.items()
    }
    live_predictions = (
        next(iter(live_predictions_by_model.values()))
        if len(live_predictions_by_model) == 1
        else _combine_predictions(live_predictions_by_model, dataset.live["era"])
    )
    live_predictions.to_frame().to_csv(run_dir / "live_predictions.csv")

    live_for_neutralization = dataset.live[["era"] + neutralizers].copy()
    live_for_neutralization["prediction"] = live_predictions
    live_predictions_neutralized = neutralize_predictions(
        live_for_neutralization, neutralizers, config.neutralization_proportion
    )
    live_predictions_neutralized = rank_normalize(live_predictions_neutralized)
    live_predictions_neutralized.to_frame().to_csv(run_dir / "live_predictions_neutralized.csv")

    validate_predictions(live_predictions_neutralized, dataset.live["era"])

    return PipelineResult(
        run_id=run_id,
        run_dir=run_dir,
        validation_scores=validation_scores,
        combined_validation_score=combined_validation_score,
        live_predictions=live_predictions,
        live_predictions_neutralized=live_predictions_neutralized,
    )


def _write_run_config(
    run_dir: Path,
    config: PipelineConfig,
    *,
    trainers: dict[str, Trainer],
    neutralizers: list[str] | None,
) -> None:
    """Record what produced this run's scores, so a comparison is attributable."""
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "models": sorted(trainers),
                "neutralizer_count": len(neutralizers) if neutralizers else 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _last_eras(df: pd.DataFrame, n: int) -> pd.DataFrame:
    eras = sorted(df["era"].unique(), key=int)[-n:]
    return df[df["era"].isin(eras)]


def _write_score(run_dir: Path, score: ValidationScore, *, prefix: str) -> None:
    (run_dir / f"{prefix}validation_score.json").write_text(
        json.dumps(
            {
                "mean_corr": score.mean_corr,
                "std_corr": score.std_corr,
                "sharpe": score.sharpe,
                "smart_sharpe": score.smart_sharpe,
            },
            indent=2,
        )
    )
    score.era_corr.to_csv(run_dir / f"{prefix}validation_era_corr.csv", header=["corr"])
