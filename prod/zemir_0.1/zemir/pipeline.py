from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI

from zemir.data import download
from zemir.scoring import (
    ValidationScore,
    neutralize_predictions,
    rank_normalize,
    score_validation,
    validate_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "prod" / "zemir_0.1" / "runs"

Trainer = Callable[[pd.DataFrame, pd.Series, pd.Series], object]


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
    validation_scores: dict[str, ValidationScore]
    combined_validation_score: ValidationScore
    live_predictions: pd.Series
    live_predictions_neutralized: pd.Series
    submission: SubmissionResult


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


def _submit_predictions(
    predictions: pd.Series, *, model_slot: str, napi: NumerAPI | None = None
) -> SubmissionResult:
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
    return SubmissionResult(model_name=model_slot, model_id=model_id, submission_id=submission_id)


def run_pipeline(
    *,
    run_id: str,
    data_version: str,
    feature_set: str,
    neutralization_proportion: float,
    min_validation_mean_corr: float,
    submission_model_slot: str,
    trainers: dict[str, Trainer],
    neutralizers: list[str] | None = None,
) -> PipelineResult:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = download(data_version, feature_set)
    feature_columns = dataset.feature_columns
    neutralizers = neutralizers or feature_columns

    train_df = dataset.train.dropna(subset=["target"])
    validation_df = dataset.validation.dropna(subset=["target"])

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

    if combined_validation_score.mean_corr < min_validation_mean_corr:
        raise ValidationScoreBelowThreshold(
            f"validation mean_corr {combined_validation_score.mean_corr:.4f} < "
            f"min_validation_mean_corr {min_validation_mean_corr:.4f} — "
            "stopping before neutralization/submission"
        )

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
        live_for_neutralization, neutralizers, neutralization_proportion
    )
    live_predictions_neutralized = rank_normalize(live_predictions_neutralized)
    live_predictions_neutralized.to_frame().to_csv(run_dir / "live_predictions_neutralized.csv")

    validate_predictions(live_predictions_neutralized, dataset.live["era"])

    submission = _submit_predictions(
        live_predictions_neutralized.rename("prediction"), model_slot=submission_model_slot
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

    return PipelineResult(
        run_id=run_id,
        validation_scores=validation_scores,
        combined_validation_score=combined_validation_score,
        live_predictions=live_predictions,
        live_predictions_neutralized=live_predictions_neutralized,
        submission=submission,
    )


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
