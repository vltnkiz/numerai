"""scripts/run_pipeline.py's ordering by role (issue #97), driven end to end on synthetic data.

Every network and scheduling seam the script reaches is swapped for a fake,
and the upload is a recorder, so nothing here can submit. The pipeline itself,
the gate, the post-submission scoring and the score log run for real.
"""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import zemir.pipeline as pipeline_module
from zemir.pipeline import SubmissionResult
from zemir.schedule import Round
from zemir.strategy import BlendSpec, Neutralization, Strategy

from factories import FEATURE_SETS, SIGNAL_COLUMN, make_dataset, predict_column_spec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pipeline_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def live(tmp_path, monkeypatch, fake_trainers):
    """The script, with every seam faked. Returns (script, events, log_path)."""
    script = _load_script()
    events: list[str] = []
    dataset = make_dataset()
    now = datetime.now(timezone.utc)
    this_round = Round(number=1, open_time=now, close_staking_time=now + timedelta(hours=1))
    log_path = tmp_path / "score_log.jsonl"

    def submit(predictions, *, model_slot, run_dir):
        events.append("submit")
        return SubmissionResult(model_name=model_slot, model_id="model-id", submission_id="sub-1")

    def load_meta_model(version, *, refresh=False, **_):
        events.append(f"load_meta_model(refresh={refresh})")
        rng = np.random.default_rng(11)
        return pd.Series(rng.normal(size=len(dataset.validation)), index=dataset.validation.index)

    def score_run(*args, **kwargs):
        events.append("score_run")
        return pipeline_module.score_run(*args, **kwargs)

    monkeypatch.setattr(script, "NumerAPI", lambda: None)
    monkeypatch.setattr(script, "round_to_run", lambda **_: this_round)
    monkeypatch.setattr(script, "fetch_current_round", lambda napi: this_round)
    monkeypatch.setattr(script, "require_available_memory", lambda gib: None)
    monkeypatch.setattr(
        script, "resolve_feature_sets", lambda version, names: {n: FEATURE_SETS[n] for n in names}
    )
    monkeypatch.setattr(script, "download", lambda *a, **k: dataset)
    monkeypatch.setattr(
        script,
        "PRODUCTION_STRATEGY",
        Strategy(
            models=(predict_column_spec("linear", "medium", SIGNAL_COLUMN),),
            blend=BlendSpec(neutralization=Neutralization(0.5, "medium")),
        ),
    )
    monkeypatch.setattr(script, "run_pipeline", partial(script.run_pipeline, runs_dir=tmp_path))
    monkeypatch.setattr(script, "submit_predictions", submit)
    monkeypatch.setattr(script, "record_round_outcome", lambda *a, **k: events.append("record_outcome"))
    monkeypatch.setattr(script, "load_meta_model", load_meta_model)
    monkeypatch.setattr(script, "score_run", score_run)
    monkeypatch.setattr(script, "refresh_validation", lambda version: events.append("refresh_validation"))
    monkeypatch.setattr(script, "append_score_log", partial(script.append_score_log, log_path=log_path))
    monkeypatch.setattr(script, "load_numerai_models", lambda: {"zemir_01": "model-id"})
    monkeypatch.setattr(
        script,
        "update_record",
        lambda models: events.append(f"update_live_scores({sorted(models)})") or [],
    )
    return script, events, log_path


def test_everything_recorded_only_runs_after_the_upload(live):
    script, events, log_path = live

    assert script.main() == 0

    assert events == [
        "submit",
        "record_outcome",
        "load_meta_model(refresh=True)",
        "score_run",
        "update_live_scores(['zemir_01'])",
        "refresh_validation",
    ]
    entry = json.loads(log_path.read_text())
    assert entry["submission_id"] == "sub-1"
    assert list(entry["models"]) == ["linear"]
    assert entry["submitted_blend"]["paid"]["mean_corr"] == pytest.approx(entry["gate"]["mean_corr"])


def test_a_failure_after_the_upload_keeps_the_log_entry_and_exits_6(live, monkeypatch, capsys):
    script, events, log_path = live

    def broken(version, **_):
        raise FileNotFoundError("meta_model.parquet: no cached copy and the refresh failed")

    monkeypatch.setattr(script, "load_meta_model", broken)

    assert script.main() == script.EXIT_POST_SUBMISSION_SCORING_FAILED

    out = capsys.readouterr()
    assert "post-submission scoring failed" in out.out
    assert "submission itself went through (submission_id=sub-1)" in out.out
    assert "FileNotFoundError" in out.err
    entry = json.loads(log_path.read_text())
    assert entry["submission_id"] == "sub-1"
    assert entry["models"] is None and entry["submitted_blend"] is None
    assert entry["gate"]["passed"] is True
    run_dir = next(p for p in log_path.parent.iterdir() if p.is_dir())
    assert json.loads((run_dir / "gate.json").read_text())["passed"] is True


def test_a_gate_failure_still_exits_2_and_is_logged_before_any_upload(live, monkeypatch):
    script, events, log_path = live
    monkeypatch.setattr(script, "MIN_VALIDATION_MEAN_CORR", 2.0)

    assert script.main() == script.EXIT_GATE_FAILED

    assert "submit" not in events
    assert events[-1] == "refresh_validation"
    entry = json.loads(log_path.read_text())
    assert entry["submission_id"] is None
    assert entry["gate"]["passed"] is False and entry["gate"]["threshold"] == 2.0
    assert entry["raw_blend"] is not None


def test_a_failed_live_score_fetch_never_changes_the_runs_outcome(live, monkeypatch, capsys):
    script, events, log_path = live

    def unreachable(models):
        raise ConnectionError("api-tournament.numer.ai unreachable")

    monkeypatch.setattr(script, "update_record", unreachable)

    assert script.main() == 0

    assert "submit" in events
    assert "fetching Numerai's live scores failed" in capsys.readouterr().out
    assert json.loads(log_path.read_text())["submission_id"] == "sub-1"


def test_a_gate_failure_still_fetches_the_live_scores(live, monkeypatch):
    script, events, log_path = live
    monkeypatch.setattr(script, "MIN_VALIDATION_MEAN_CORR", 2.0)

    assert script.main() == script.EXIT_GATE_FAILED

    assert "update_live_scores(['zemir_01'])" in events


def test_a_round_change_skips_the_validation_refresh(live, monkeypatch):
    """The wrapper reruns at once for the new round; a 5.6 GB download must not delay it."""
    script, events, log_path = live
    new_round = Round(number=2, open_time=datetime.now(timezone.utc), close_staking_time=None)
    monkeypatch.setattr(script, "fetch_current_round", lambda napi: new_round)

    assert script.main() == script.EXIT_ROUND_CHANGED

    assert "submit" not in events and "refresh_validation" not in events
