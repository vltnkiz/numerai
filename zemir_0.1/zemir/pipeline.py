"""Pipeline entrypoint: wire download -> train -> neutralize -> submit together.

Stages exchange data as in-memory DataFrames within this one process, per
docs/adr/0001-zemir-pipeline-architecture.md; each stage's own output is also
persisted under `runs/<run_id>/` here, as the side effect the ADR calls for.

Submission is gated on the validation score computed by the training stage
(`zemir/train.py`) — the threshold/hard-stop decision docs/decisions #7 left
open for this ticket: if `validation_score.mean_corr` falls below
`RunConfig.min_validation_mean_corr`, the run stops before neutralizing or
submitting anything.

No separate round-open gate: ticket #10 found `NumerAPI.check_round_open()`
reported the round closed on a run where `upload_predictions` nonetheless
succeeded, so it's not a reliable pre-check. Instead, ticket #11's scheduled
GitHub Actions job fires generously across Numerai's Tuesday-Saturday window
(docs/research/live-round-data-and-submission.md) and simply lets this
function run every time — a genuinely closed round is expected to surface as
a failed run via whatever exception `submit_predictions` raises, not a
silent no-op here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zemir.config import RunConfig
from zemir.download import download
from zemir.models.base import Model
from zemir.neutralize import neutralize_predictions
from zemir.submit import SubmissionResult, submit_predictions
from zemir.train import TrainResult, train_and_validate

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "zemir_0.1" / "runs"


class ValidationScoreBelowThreshold(RuntimeError):
    """Raised when a run's validation score doesn't clear `min_validation_mean_corr`."""


@dataclass
class PipelineResult:
    run_id: str
    train_result: TrainResult
    live_predictions: pd.Series
    live_predictions_neutralized: pd.Series
    submissions: list[SubmissionResult]


def _run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_pipeline(config: RunConfig, model: Model) -> PipelineResult:
    """Run one end-to-end pipeline invocation: download, train, neutralize, submit.

    `model` is caller-supplied (per docs/adr/0001) so this function is
    identical for linear regression and, later, era-boosted XGBoost.
    """
    run_dir = _run_dir(config.run_id)

    download_result = download(config.data_version, feature_set=config.features.feature_set)
    feature_columns = download_result.feature_sets[config.features.feature_set]
    neutralizers = config.features.neutralizers or feature_columns

    train_result = train_and_validate(
        model, download_result.train, download_result.validation, feature_columns
    )
    _write_validation_score(run_dir, train_result)

    if train_result.validation_score.mean_corr < config.min_validation_mean_corr:
        raise ValidationScoreBelowThreshold(
            f"validation mean_corr {train_result.validation_score.mean_corr:.4f} < "
            f"min_validation_mean_corr {config.min_validation_mean_corr:.4f} — "
            "stopping before neutralization/submission"
        )

    live_predictions = pd.Series(
        model.predict(download_result.live[feature_columns]),
        index=download_result.live.index,
        name="prediction",
    )
    live_predictions.to_frame().to_csv(run_dir / "live_predictions.csv")

    live_for_neutralization = download_result.live[["era"] + neutralizers].copy()
    live_for_neutralization["prediction"] = live_predictions
    live_predictions_neutralized = neutralize_predictions(
        live_for_neutralization, neutralizers, config.neutralization.proportion
    )
    live_predictions_neutralized.to_frame().to_csv(
        run_dir / "live_predictions_neutralized.csv"
    )

    submissions = submit_predictions(live_predictions_neutralized.rename("prediction"))
    _write_submissions(run_dir, submissions)

    return PipelineResult(
        run_id=config.run_id,
        train_result=train_result,
        live_predictions=live_predictions,
        live_predictions_neutralized=live_predictions_neutralized,
        submissions=submissions,
    )


def _write_validation_score(run_dir: Path, train_result: TrainResult) -> None:
    score = train_result.validation_score
    (run_dir / "validation_score.json").write_text(
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
    score.era_corr.to_csv(run_dir / "validation_era_corr.csv", header=["corr"])


def _write_submissions(run_dir: Path, submissions: list[SubmissionResult]) -> None:
    (run_dir / "submissions.json").write_text(
        json.dumps(
            [
                {
                    "model_name": s.model_name,
                    "model_id": s.model_id,
                    "submission_id": s.submission_id,
                }
                for s in submissions
            ],
            indent=2,
        )
    )
