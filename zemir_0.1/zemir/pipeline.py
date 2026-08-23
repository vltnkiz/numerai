"""Pipeline entrypoint: wire download -> train -> ensemble -> neutralize -> submit.

Stages exchange data as in-memory DataFrames within this one process, per
docs/adr/0001-zemir-pipeline-architecture.md; each stage's own output is also
persisted under `runs/<run_id>/` here, as the side effect the ADR calls for.

`models` is a name -> Model mapping rather than a single model so more than
one model type can run in the same invocation and have their predictions
combined (zemir/ensemble.py) — [Ensembling strategy]
(https://github.com/vltnkiz/numerai/issues/14). A single-model run is just
`models` with one entry and no `RunConfig.ensemble` needed.

Submission is gated on the *combined* validation score — the threshold/
hard-stop decision docs/decisions #7 left open for ticket #10: if
`validation_score.mean_corr` falls below `RunConfig.min_validation_mean_corr`,
the run stops before neutralizing or submitting anything.

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

from zemir.config import EnsembleConfig, RunConfig
from zemir.download import download
from zemir.ensemble import combine_predictions
from zemir.metrics import ValidationScore, score_validation
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
    train_results: dict[str, TrainResult]
    validation_score: ValidationScore
    live_predictions: pd.Series
    live_predictions_neutralized: pd.Series
    submissions: list[SubmissionResult]


def _run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_weights(
    ensemble_config: EnsembleConfig | None, models: dict[str, Model]
) -> dict[str, float]:
    if len(models) == 1:
        return {next(iter(models)): 1.0}
    if ensemble_config is None:
        raise ValueError(
            f"multiple models given ({sorted(models)}) but RunConfig.ensemble is "
            "None — set ensemble weights for each model"
        )
    if models.keys() != ensemble_config.weights.keys():
        raise ValueError(
            f"RunConfig.ensemble.weights {sorted(ensemble_config.weights)} must "
            f"name the same models as `models` {sorted(models)}"
        )
    return ensemble_config.weights


def run_pipeline(config: RunConfig, models: dict[str, Model]) -> PipelineResult:
    """Run one end-to-end pipeline invocation: download, train, ensemble, neutralize, submit.

    `models` is caller-supplied (per docs/adr/0001) so this function is
    identical whether it's running one model or combining several.
    """
    run_dir = _run_dir(config.run_id)
    weights = _resolve_weights(config.ensemble, models)

    download_result = download(config.data_version, feature_set=config.features.feature_set)
    feature_columns = download_result.feature_sets[config.features.feature_set]
    neutralizers = config.features.neutralizers or feature_columns

    train_results = {
        name: train_and_validate(
            model, download_result.train, download_result.validation, feature_columns
        )
        for name, model in models.items()
    }
    _write_validation_scores(run_dir, train_results)

    validation_score = _combined_validation_score(download_result.validation, train_results, weights)
    _write_combined_validation_score(run_dir, validation_score)

    if validation_score.mean_corr < config.min_validation_mean_corr:
        raise ValidationScoreBelowThreshold(
            f"validation mean_corr {validation_score.mean_corr:.4f} < "
            f"min_validation_mean_corr {config.min_validation_mean_corr:.4f} — "
            "stopping before neutralization/submission"
        )

    live_predictions_by_model = {
        name: pd.Series(
            model.predict(download_result.live[feature_columns]),
            index=download_result.live.index,
            name="prediction",
        )
        for name, model in models.items()
    }
    live_predictions = combine_predictions(live_predictions_by_model, weights)
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
        train_results=train_results,
        validation_score=validation_score,
        live_predictions=live_predictions,
        live_predictions_neutralized=live_predictions_neutralized,
        submissions=submissions,
    )


def _combined_validation_score(
    validation_df: pd.DataFrame,
    train_results: dict[str, TrainResult],
    weights: dict[str, float],
    *,
    target_col: str = "target",
    era_col: str = "era",
) -> ValidationScore:
    validation_predictions = {
        name: tr.validation_predictions for name, tr in train_results.items()
    }
    combined_predictions = combine_predictions(validation_predictions, weights)

    validation_df = validation_df.dropna(subset=[target_col])
    scored = validation_df[[era_col, target_col]].copy()
    scored["prediction"] = combined_predictions
    return score_validation(
        scored, target_col=target_col, prediction_col="prediction", era_col=era_col
    )


def _write_validation_scores(run_dir: Path, train_results: dict[str, TrainResult]) -> None:
    for name, train_result in train_results.items():
        _write_score(run_dir, train_result.validation_score, prefix=f"{name}_")


def _write_combined_validation_score(run_dir: Path, validation_score: ValidationScore) -> None:
    _write_score(run_dir, validation_score, prefix="")


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
