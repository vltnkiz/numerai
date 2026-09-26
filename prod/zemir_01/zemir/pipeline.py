from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI

from zemir.blend import blend_strategies, combine_predictions
from zemir.config import PipelineConfig
from zemir.data import Dataset, scoring_window
from zemir.fitting import fit_strategy
from zemir.models import FittedModel
from zemir.scoring import (
    EraSpearmanScore,
    PredictionScores,
    era_numerai_corr,
    era_spearman,
    rank_normalize,
    score_predictions,
    summarize_era_spearman,
    validate_predictions,
)
from zemir.strategy import FeatureSets, Strategy, neutralizer_columns, required_columns

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "prod" / "zemir_01" / "runs"
SCORE_LOG_PATH = REPO_ROOT / "prod" / "zemir_01" / "score_log.jsonl"

# Issue #68: how long a live run's scores stay in the committed history log.
# Run directories on disk (gitignored, ~100 MB/run) are kept indefinitely —
# the machine running the live job has hundreds of GB free, so pruning them
# isn't worth the code.
RETENTION_DAYS = 365
# Issue #98: the history log's entry shape. Schema 1 (#68) carries no marker.
SCORE_LOG_SCHEMA = 2
# The paid row's two counts, recorded as integers rather than floats.
_INTEGER_SCORE_COLUMNS = {"eras", "mmc_eras"}
_RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"

# The two blends a live run scores, as columns beside each model's (issue #97).
# `combined` is the raw blend, never submitted, under the name schema 1 of the
# history log used for it (schema 2 says `raw_blend`, issue #98);
# `combined_neutralized` is the submitted blend, the one the gate reads and the
# one a harness row measures. See CONTEXT.md.
RAW_BLEND = "combined"
SUBMITTED_BLEND = "combined_neutralized"


@dataclass
class SubmissionResult:
    model_name: str
    model_id: str
    submission_id: str


@dataclass
class PipelineResult:
    """What one live run produced, up to the gate and no further.

    Carries the gate's one number and the predictions everything recorded-only
    is scored from, but no other score: scoring that nothing waits on runs
    after the submission (`score_run`), so its cost can never delay or break
    one (issue #97).

    `validation_predictions` holds one column per model plus `RAW_BLEND` and
    `SUBMITTED_BLEND`, on `validation`'s index. `gate_corr` is the submitted
    blend's mean **Numerai CORR** over the whole of `validation`.
    """

    run_id: str
    run_dir: Path
    validation: pd.DataFrame
    validation_predictions: pd.DataFrame
    gate_corr: float
    live_predictions: pd.Series
    live_predictions_neutralized: pd.Series


@dataclass(frozen=True)
class RunScores:
    """Everything recorded about a run after its submission.

    `paid` is `score_predictions`' table over every column of
    `PipelineResult.validation_predictions`: the composition, and the files, a
    harness scoring run produces. `spearman` holds the legacy **Spearman
    correlation** triple for each model and the raw blend, kept only so the
    history log's older entries stay comparable (issue #96).
    """

    paid: PredictionScores
    spearman: dict[str, EraSpearmanScore]


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
    scripts/run_pipeline.py — the entrypoint the scheduled task invokes —
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


def predict_each(
    fitted_models: Mapping[str, FittedModel], df: pd.DataFrame
) -> dict[str, pd.Series]:
    """Each model's predictions on `df`, each drawn from its own columns."""
    return {
        name: pd.Series(model.predict(df), index=df.index, name="prediction")
        for name, model in fitted_models.items()
    }


def feature_set_names(config: PipelineConfig, strategy: Strategy) -> list[str]:
    """Every feature set a live run resolves: its scoring universe, and the strategy's own."""
    return list(dict.fromkeys([config.feature_set, *strategy.feature_set_names]))


def run_columns(config: PipelineConfig, strategy: Strategy, feature_sets: FeatureSets) -> list[str]:
    """What a live run loads: what the strategy needs, plus the run's scoring universe.

    `config.feature_set` is the universe feature exposure is measured against,
    exactly as the harness measures it against its cache's `feature_set`
    (CONTEXT.md, "Feature set"). A strategy that already loads all of it (as
    production does) loads the same columns as before.
    """
    return list(
        dict.fromkeys([*required_columns(strategy, feature_sets), *feature_sets[config.feature_set]])
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    run_id: str,
    strategy: Strategy,
    feature_sets: FeatureSets,
    dataset: Dataset,
    runs_dir: Path = RUNS_DIR,
) -> PipelineResult:
    """Train, score, predict, neutralize and write artifacts.

    Takes `dataset` rather than downloading one itself — the caller decides
    where it comes from (`zemir.data.download` in production, a synthetic
    fixture in a test), so the whole post-fit path below is exercisable in
    milliseconds without a network call.

    `dataset` must be loaded at (at least) the width the strategy needs —
    `zemir.strategy.required_columns` — since each model is fitted on its own
    slice of it (`zemir.fitting`, which owns the memory discipline of the fit;
    `dataset.train` stays the caller's, resident for the whole run, exactly as
    before) and every neutralization projects onto columns it names.
    `feature_sets` resolves each feature-set name, on a model or a
    neutralization, to columns.

    `live_predictions.csv` is the raw blend, before any neutralization at all;
    `live_predictions_neutralized.csv` is what `zemir.blend` returns, ranked.

    The submitted blend is also built on validation, from `float32` model
    predictions and cast to `float32` after, exactly as `zemir.harness` builds
    it from its cache, so its row is the harness's row to the digit.
    `gate_corr` is its mean Numerai CORR. Nothing else is scored here.

    Never submits and never gates: submission is a separate step
    (`submit_predictions`), and the gate that guards it (`record_gate`) is
    called beside it in scripts/run_pipeline.py.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    blend = strategy.blend

    _write_run_config(run_dir, config, strategy=strategy, feature_sets=feature_sets)

    fit = fit_strategy(
        strategy,
        lambda: dataset.train,
        feature_sets=feature_sets,
        max_eras=config.max_eras,
    )
    fitted_models = fit.models
    validation_df = scoring_window(dataset.validation, config.max_eras)
    validation_by_model = predict_each(fitted_models, validation_df)
    validation_predictions = pd.DataFrame(validation_by_model)
    validation_predictions[RAW_BLEND] = combine_predictions(
        validation_by_model, validation_df["era"], blend.weights
    )
    validation_predictions[SUBMITTED_BLEND] = blend_strategies(
        {"live": strategy},
        pd.DataFrame(validation_by_model).astype("float32"),
        validation_df,
        validation_df["era"],
        feature_sets,
    )["live"].astype("float32")
    gate_corr = float(
        era_numerai_corr(
            validation_predictions[[SUBMITTED_BLEND]],
            validation_df["target"],
            validation_df["era"],
        )[SUBMITTED_BLEND].mean()
    )

    live_predictions_by_model = predict_each(fitted_models, dataset.live)
    live_predictions = combine_predictions(
        live_predictions_by_model, dataset.live["era"], blend.weights
    )
    live_predictions.to_frame().to_csv(run_dir / "live_predictions.csv")

    # The same transform the harness scores (`zemir.blend`), here as the one-strategy
    # case: per-model neutralization, the blend, then the blend's neutralization.
    live_predictions_neutralized = blend_strategies(
        {"live": strategy},
        pd.DataFrame(live_predictions_by_model),
        dataset.live,
        dataset.live["era"],
        feature_sets,
    )["live"].rename("prediction_neutralized")
    live_predictions_neutralized = rank_normalize(live_predictions_neutralized)
    live_predictions_neutralized.to_frame().to_csv(run_dir / "live_predictions_neutralized.csv")

    validate_predictions(live_predictions_neutralized, dataset.live["era"])

    return PipelineResult(
        run_id=run_id,
        run_dir=run_dir,
        validation=validation_df,
        validation_predictions=validation_predictions,
        gate_corr=gate_corr,
        live_predictions=live_predictions,
        live_predictions_neutralized=live_predictions_neutralized,
    )


def record_gate(result: PipelineResult, *, threshold: float) -> dict:
    """The gate's record, written to `gate.json` before any upload, and returned.

    `passed` is exactly `gate_corr >= threshold`. Written first, so the gate's
    evidence survives whatever happens after it: a failed upload, or a failed
    post-submission score. The same record becomes the history log entry's
    `gate` (issue #98), so the two never describe the gate in different shapes.
    """
    record = {
        "metric": "numerai_corr",
        "artifact": SUBMITTED_BLEND,
        "eras": int(result.validation["era"].nunique()),
        "mean_corr": result.gate_corr,
        "threshold": threshold,
        "passed": not result.gate_corr < threshold,
    }
    (result.run_dir / "gate.json").write_text(json.dumps(record, indent=2))
    return record


def score_run(
    result: PipelineResult,
    *,
    meta_model: pd.Series,
    scoring_universe: list[str],
) -> RunScores:
    """Score every validation column and write the artifacts. Runs after the submission.

    The paid table goes through `score_predictions` over a `float32` frame:
    the harness's composition, dtype and filenames, so a live run directory's
    scores read like a harness cache's. The Spearman triple is taken on the
    unconverted predictions, the inputs it has always had, so the history log's
    Spearman series runs unbroken across its schema 1 and schema 2 entries.
    `scoring_universe` is what exposure is measured against: the run's
    `feature_set`, never a strategy's own columns.
    """
    validation = result.validation
    predictions = result.validation_predictions
    paid = score_predictions(
        predictions.astype("float32"),
        validation["target"],
        validation["era"],
        meta_model=meta_model,
        features=validation[scoring_universe],
    )
    spearman = {
        name: summarize_era_spearman(
            era_spearman(validation[["era", "target"]].assign(prediction=predictions[name]))
        )
        for name in predictions.columns
        if name != SUBMITTED_BLEND
    }

    run_dir = result.run_dir
    # The harness's filenames (`zemir.harness.score_configs`), column for column.
    paid.summary.to_csv(run_dir / "scores.csv")
    paid.corr_by_era.to_csv(run_dir / "era_corr.csv")
    paid.mmc_by_era.to_csv(run_dir / "era_mmc.csv")
    paid.exposure_by_era.to_csv(run_dir / "era_max_feature_corr.csv")
    pd.DataFrame(
        {
            name: {
                "mean_corr": score.mean_corr,
                "std_corr": score.std_corr,
                "sharpe": score.sharpe,
                "smart_sharpe": score.smart_sharpe,
            }
            for name, score in spearman.items()
        }
    ).T.rename_axis("config").to_csv(run_dir / "spearman_scores.csv")
    pd.DataFrame({name: score.era_spearman for name, score in spearman.items()}).rename_axis(
        "era"
    ).to_csv(run_dir / "era_spearman.csv")
    return RunScores(paid=paid, spearman=spearman)


# The paid columns a run prints. `mmc_eras` rides beside MMC and payout on
# purpose: both are only comparable between runs scored over the same
# meta-model window, and that window grows about one era a week (issue #101).
_PRINTED_COLUMNS = ["eras", "mean_corr", "sharpe", "mmc_eras", "mean_mmc", "payout", "max_feature_corr"]


def format_gate(result: PipelineResult, *, threshold: float) -> str:
    """The gate's one line: which number, on which artifact, against which floor."""
    verdict = "pass" if not result.gate_corr < threshold else "FAIL"
    return (
        f"gate: numerai_corr({SUBMITTED_BLEND}) = {result.gate_corr:.6f}, "
        f"floor MIN_VALIDATION_MEAN_CORR = {threshold:.4f}: {verdict}"
    )


def format_run_scores(scores: RunScores) -> str:
    """The paid table, then the raw blend's legacy Spearman line, labelled as legacy."""
    legacy = scores.spearman[RAW_BLEND]
    return "\n".join(
        [
            "validation, Numerai CORR / MMC (the harness's vocabulary):",
            scores.paid.summary[_PRINTED_COLUMNS].to_string(float_format=lambda v: f"{v:.6f}"),
            f"legacy Spearman, {RAW_BLEND}: mean_corr={legacy.mean_corr:.4f}  "
            f"sharpe={legacy.sharpe:.4f}  smart_sharpe={legacy.smart_sharpe:.4f}",
        ]
    )


def _write_run_config(
    run_dir: Path,
    config: PipelineConfig,
    *,
    strategy: Strategy,
    feature_sets: FeatureSets,
) -> None:
    """Record what produced this run's scores, so a comparison is attributable."""
    blend = strategy.blend
    blend_neutralization = blend.neutralization
    record = {
        **asdict(config),
        "models": sorted(spec.name for spec in strategy.models),
        "model_features": {spec.name: spec.features for spec in strategy.models},
        "model_weights": dict(blend.weights) if blend.weights is not None else None,
        "neutralization_proportion": (
            blend_neutralization.proportion if blend_neutralization is not None else 0.0
        ),
        "neutralizer_count": (
            len(neutralizer_columns(blend_neutralization, feature_sets))
            if blend_neutralization is not None
            else 0
        ),
    }
    # Written only when a model neutralizes: the key records what was applied, and
    # a strategy that applies nothing per model keeps the record it always had.
    per_model = {
        spec.name: {
            "proportion": spec.neutralization.proportion,
            "neutralizer_count": len(neutralizer_columns(spec.neutralization, feature_sets)),
        }
        for spec in strategy.models
        if spec.neutralization is not None
    }
    if per_model:
        record["model_neutralization"] = per_model
    (run_dir / "config.json").write_text(json.dumps(record, indent=2, sort_keys=True))


def _run_id_age_days(run_id: str, *, now: datetime) -> float | None:
    """Days between `run_id`'s timestamp and `now`, or None if `run_id` isn't one."""
    try:
        stamp = datetime.strptime(run_id, _RUN_ID_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (now - stamp).total_seconds() / 86400


def append_score_log(
    *,
    run_id: str,
    target_column: str,
    gate: Mapping[str, object],
    scores: RunScores | None,
    submission_id: str | None,
    log_path: Path = SCORE_LOG_PATH,
    now: datetime | None = None,
) -> None:
    """Append one live run's `SCORE_LOG_SCHEMA` entry to the cross-run history log.

    Called for every scripts/run_pipeline.py invocation regardless of the
    gate, so a run that fails the gate still leaves a trace — the gate only
    catches a run crashing through the floor, not one quietly declining above
    it (issue #68). Prunes entries older than `RETENTION_DAYS` on each append,
    so the file stays small forever rather than growing without bound. The
    prune reads nothing but `run_id`, so it spans every schema.

    `gate` is `record_gate`'s record, known before the upload. `scores` is
    `score_run`'s; `None` means post-submission scoring failed, and the entry
    is still written with its three artifacts `null`, so neither the run nor
    its submission ever goes missing from the log (issue #97).

    Schema 2 (issue #98) nests by artifact — `models`, `raw_blend`,
    `submitted_blend` — and within each by vocabulary: `paid` is the
    artifact's full harness row, `mmc_eras` included, so MMC and payout carry
    their window; `spearman` is the legacy triple, present only where schema 1
    recorded one. An entry with no `schema` key is schema 1:
    `{run_id, target_column, models, combined, submission_id}`, Spearman only,
    `combined` being the raw blend. Those are left exactly as written.
    """

    def _paid(name: str) -> dict[str, float | int]:
        row = scores.paid.summary.loc[name]
        return {k: int(v) if k in _INTEGER_SCORE_COLUMNS else float(v) for k, v in row.items()}

    def _spearman(name: str) -> dict[str, float]:
        s = scores.spearman[name]
        return {"mean_corr": s.mean_corr, "sharpe": s.sharpe, "smart_sharpe": s.smart_sharpe}

    now = now or datetime.now(timezone.utc)
    entries = []
    if log_path.exists():
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    entries = [
        e
        for e in entries
        if (age := _run_id_age_days(e["run_id"], now=now)) is not None and age <= RETENTION_DAYS
    ]
    models = [name for name in scores.spearman if name != RAW_BLEND] if scores else []
    entries.append(
        {
            "schema": SCORE_LOG_SCHEMA,
            "run_id": run_id,
            "target_column": target_column,
            "gate": dict(gate),
            "models": (
                {name: {"paid": _paid(name), "spearman": _spearman(name)} for name in models}
                if scores
                else None
            ),
            "raw_blend": (
                {"paid": _paid(RAW_BLEND), "spearman": _spearman(RAW_BLEND)} if scores else None
            ),
            "submitted_blend": {"paid": _paid(SUBMITTED_BLEND)} if scores else None,
            "submission_id": submission_id,
        }
    )
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
