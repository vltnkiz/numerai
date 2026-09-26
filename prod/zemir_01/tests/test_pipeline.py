import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from zemir.config import LIVE, MIN_VALIDATION_MEAN_CORR
from zemir.pipeline import (
    RAW_BLEND,
    SUBMITTED_BLEND,
    append_score_log,
    feature_set_names,
    record_gate,
    run_columns,
    run_pipeline,
    score_run,
)
from zemir.scoring import era_numerai_corr, era_spearman, summarize_era_spearman
from zemir.strategy import BlendSpec, Neutralization, Strategy, required_columns

from factories import FEATURE_SETS, SIGNAL_COLUMN, make_dataset, predict_column_spec


def _strategy(*specs, blend=BlendSpec()):
    return Strategy(models=specs, blend=blend)


def test_run_pipeline_exercises_the_post_fit_path_from_a_synthetic_dataset(tmp_path, fake_trainers):
    """Accepting a `Dataset` argument is what makes this run in milliseconds (issue #75)."""
    dataset = make_dataset()

    result = run_pipeline(
        LIVE,
        run_id="test-run",
        strategy=_strategy(predict_column_spec("linear", "medium", SIGNAL_COLUMN)),
        feature_sets=FEATURE_SETS,
        dataset=dataset,
        runs_dir=tmp_path,
    )

    assert result.run_id == "test-run"
    assert (result.run_dir / "config.json").exists()
    assert (result.run_dir / "live_predictions.csv").exists()
    assert (result.run_dir / "live_predictions_neutralized.csv").exists()
    assert list(result.validation_predictions.columns) == [
        "linear",
        RAW_BLEND,
        SUBMITTED_BLEND,
    ]
    # rank_normalize's own contract: strictly inside (0, 1), Numerai's accepted range.
    assert result.live_predictions_neutralized.min() > 0.0
    assert result.live_predictions_neutralized.max() < 1.0


def test_run_pipeline_gate_passes_when_predictions_track_the_target(tmp_path, fake_trainers):
    dataset = make_dataset()

    result = run_pipeline(
        LIVE,
        run_id="above-gate",
        strategy=_strategy(predict_column_spec("linear", "medium", SIGNAL_COLUMN)),
        feature_sets=FEATURE_SETS,
        dataset=dataset,
        runs_dir=tmp_path,
    )

    assert record_gate(result, threshold=MIN_VALIDATION_MEAN_CORR)


def test_run_pipeline_gate_fails_when_predictions_are_anti_correlated_with_the_target(
    tmp_path, fake_trainers
):
    dataset = make_dataset()

    result = run_pipeline(
        LIVE,
        run_id="below-gate",
        strategy=_strategy(
            predict_column_spec("linear", "medium", SIGNAL_COLUMN, negated=True)
        ),
        feature_sets=FEATURE_SETS,
        dataset=dataset,
        runs_dir=tmp_path,
    )

    # A sign flip: the one failure a floor of zero exists to catch (issue #95).
    assert not record_gate(result, threshold=MIN_VALIDATION_MEAN_CORR)


def test_a_weak_but_positive_fit_passes_the_gate_by_design(tmp_path, fake_trainers):
    """#95 put the floor at zero on purpose: the gate stops a sign flip, not a weak week.

    A fit whose signal is mostly drowned by noise must still pass. A test
    expecting this to fail would be asking the gate to do the harness's job.
    """
    result = run_pipeline(
        LIVE,
        run_id="weak",
        strategy=_strategy(predict_column_spec("linear", "medium", SIGNAL_COLUMN)),
        feature_sets=FEATURE_SETS,
        dataset=make_dataset(noise_scale=10.0),
        runs_dir=tmp_path,
    )

    assert 0.0 < result.gate_corr < 0.2
    assert record_gate(result, threshold=MIN_VALIDATION_MEAN_CORR)


def test_the_gate_reads_the_submitted_blend_not_the_raw_one(tmp_path, fake_trainers):
    """Neutralizing away the signal lowers `gate_corr`, while the raw blend keeps it.

    With a threshold between the two blends' scores, the run fails, which it
    could not do if the gate still read the raw blend.
    """
    strategy = _strategy(
        predict_column_spec("linear", "medium", SIGNAL_COLUMN),
        blend=BlendSpec(neutralization=Neutralization(1.0, "narrow")),
    )
    result = _run(tmp_path, strategy, "neutralized-away")
    raw_corr = era_numerai_corr(
        result.validation_predictions[[RAW_BLEND]],
        result.validation["target"],
        result.validation["era"],
    )[RAW_BLEND].mean()

    assert result.gate_corr < raw_corr
    assert not record_gate(result, threshold=(result.gate_corr + raw_corr) / 2)


def test_gate_json_is_the_gates_whole_record(tmp_path, fake_trainers):
    result = _run(tmp_path, _two_models(), "recorded-gate")

    record_gate(result, threshold=MIN_VALIDATION_MEAN_CORR)

    gate = json.loads((result.run_dir / "gate.json").read_text())
    assert gate == {
        "metric": "numerai_corr",
        "artifact": SUBMITTED_BLEND,
        "eras": 5,
        "mean_corr": result.gate_corr,
        "threshold": MIN_VALIDATION_MEAN_CORR,
        "passed": True,
    }


def test_run_pipeline_scores_nothing_but_the_gate(tmp_path, fake_trainers):
    """Everything recorded-only waits for `score_run`, after the submission (issue #97)."""
    result = _run(tmp_path, _two_models(), "pre-submission")

    written = {p.name for p in result.run_dir.iterdir()}
    assert written == {"config.json", "live_predictions.csv", "live_predictions_neutralized.csv"}


def _meta_model(result):
    rng = np.random.default_rng(11)
    return pd.Series(rng.normal(size=len(result.validation)), index=result.validation.index)


def test_score_run_writes_the_harness_files_and_the_legacy_spearman_beside_them(tmp_path, fake_trainers):
    result = _run(tmp_path, _two_models(blend_neutralization=Neutralization(0.5, "medium")), "scored")

    scores = score_run(result, meta_model=_meta_model(result), scoring_universe=FEATURE_SETS["medium"])

    for name in ("scores.csv", "era_corr.csv", "era_mmc.csv", "era_max_feature_corr.csv"):
        assert (result.run_dir / name).exists()
    assert list(scores.paid.summary.index) == ["m1", "m2", RAW_BLEND, SUBMITTED_BLEND]
    # Spearman is legacy: never measured on the submitted blend, which it never covered.
    assert list(scores.spearman) == ["m1", "m2", RAW_BLEND]
    spearman = pd.read_csv(result.run_dir / "spearman_scores.csv", index_col="config")
    assert list(spearman.index) == ["m1", "m2", RAW_BLEND]
    assert spearman.loc[RAW_BLEND, "mean_corr"] == pytest.approx(scores.spearman[RAW_BLEND].mean_corr)
    assert scores.paid.summary.loc[SUBMITTED_BLEND, "mean_corr"] == pytest.approx(result.gate_corr)


def test_score_run_keeps_the_spearman_numbers_the_old_pipeline_produced(tmp_path, fake_trainers):
    """Taken on the unconverted predictions, so the history log's series does not shift."""
    result = _run(tmp_path, _two_models(), "legacy")

    scores = score_run(result, meta_model=_meta_model(result), scoring_universe=FEATURE_SETS["medium"])

    frame = result.validation[["era", "target"]]
    for name in ("m1", "m2", RAW_BLEND):
        old = summarize_era_spearman(
            era_spearman(frame.assign(prediction=result.validation_predictions[name]))
        )
        assert scores.spearman[name].mean_corr == old.mean_corr
        assert scores.spearman[name].smart_sharpe == old.smart_sharpe


def test_the_score_log_keeps_its_shape_and_keeps_the_entry_when_scoring_failed(tmp_path, fake_trainers):
    result = _run(tmp_path, _two_models(), "logged")
    scores = score_run(result, meta_model=_meta_model(result), scoring_universe=FEATURE_SETS["medium"])
    log = tmp_path / "score_log.jsonl"
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)

    append_score_log(
        run_id="20260926T120000Z", target_column="target", spearman=scores.spearman,
        submission_id="sub-1", log_path=log, now=now,
    )
    append_score_log(
        run_id="20260926T130000Z", target_column="target", spearman=None,
        submission_id="sub-2", log_path=log, now=now,
    )

    scored, failed = [json.loads(line) for line in log.read_text().splitlines()]
    assert list(scored) == ["run_id", "target_column", "models", "combined", "submission_id"]
    assert list(scored["models"]) == ["m1", "m2"]
    assert list(scored["combined"]) == ["mean_corr", "sharpe", "smart_sharpe"]
    assert failed["models"] is None and failed["combined"] is None
    assert failed["submission_id"] == "sub-2"


def test_run_columns_adds_the_scoring_universe_and_nothing_for_production_shaped_strategies():
    narrow_only = _strategy(predict_column_spec("n", "narrow", "feature_a"))
    medium = _strategy(predict_column_spec("m", "medium", "feature_a"))

    assert feature_set_names(LIVE, narrow_only) == ["medium", "narrow"]
    assert run_columns(LIVE, narrow_only, FEATURE_SETS) == ["feature_a", "feature_b", "feature_c"]
    assert run_columns(LIVE, medium, FEATURE_SETS) == required_columns(medium, FEATURE_SETS)


def test_run_pipeline_fits_two_specs_at_different_widths_in_one_run(tmp_path, fake_trainers):
    """#80's done-when, end to end: the wide model reads a column the narrow one never sees."""
    dataset = make_dataset()
    strategy = _strategy(
        predict_column_spec("narrow_model", "narrow", "feature_a"),
        predict_column_spec("wide_model", "medium", "feature_c"),
    )

    result = run_pipeline(
        LIVE,
        run_id="two-widths",
        strategy=strategy,
        feature_sets=FEATURE_SETS,
        dataset=dataset,
        runs_dir=tmp_path,
    )

    assert {"narrow_model", "wide_model"} <= set(result.validation_predictions.columns)
    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["model_features"] == {"narrow_model": "narrow", "wide_model": "medium"}


def _two_models(m2_neutralization=None, blend_neutralization=None):
    specs = (
        predict_column_spec("m1", "medium", "feature_a"),
        predict_column_spec("m2", "medium", "feature_c"),
    )
    if m2_neutralization is not None:
        specs = (specs[0], replace(specs[1], neutralization=m2_neutralization))
    return _strategy(*specs, blend=BlendSpec(neutralization=blend_neutralization))


def _run(tmp_path, strategy, run_id):
    return run_pipeline(
        LIVE,
        run_id=run_id,
        strategy=strategy,
        feature_sets=FEATURE_SETS,
        dataset=make_dataset(),
        runs_dir=tmp_path,
    )


def test_run_pipeline_neutralizes_one_model_before_the_blend(tmp_path, fake_trainers):
    """`ModelSpec.neutralization` has a reader: the live artifact moves when it is set (#82's evidence, #88's fix)."""
    plain = _run(tmp_path, _two_models(), "plain")
    neutralized = _run(tmp_path, _two_models(m2_neutralization=Neutralization(1.0, "narrow")), "per-model")

    assert not plain.live_predictions_neutralized.equals(neutralized.live_predictions_neutralized)


def test_live_predictions_csv_is_the_raw_blend_whatever_is_neutralized(tmp_path, fake_trainers):
    plain = _run(tmp_path, _two_models(), "plain")
    neutralized = _run(
        tmp_path,
        _two_models(
            m2_neutralization=Neutralization(1.0, "narrow"),
            blend_neutralization=Neutralization(0.5, "medium"),
        ),
        "all-stages",
    )

    assert (plain.run_dir / "live_predictions.csv").read_bytes() == (
        neutralized.run_dir / "live_predictions.csv"
    ).read_bytes()


def test_the_neutralized_artifact_keeps_the_column_name_it_has_always_had(tmp_path, fake_trainers):
    result = _run(tmp_path, _two_models(blend_neutralization=Neutralization(0.5, "medium")), "named")

    header = (result.run_dir / "live_predictions_neutralized.csv").read_text().splitlines()[0]
    assert header == "id,prediction_neutralized"


def test_config_json_records_the_blend_neutralization_in_the_keys_it_always_had(tmp_path, fake_trainers):
    result = _run(tmp_path, _two_models(blend_neutralization=Neutralization(0.95, ("feature_a", "feature_b"))), "keys")

    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["neutralization_proportion"] == 0.95
    assert config["neutralizer_count"] == 2
    assert "model_neutralization" not in config


def test_config_json_records_a_per_model_neutralization_only_when_there_is_one(tmp_path, fake_trainers):
    result = _run(tmp_path, _two_models(m2_neutralization=Neutralization(0.4, "narrow")), "recorded")

    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["model_neutralization"] == {"m2": {"proportion": 0.4, "neutralizer_count": 1}}
    assert config["neutralization_proportion"] == 0.0
    assert config["neutralizer_count"] == 0


def test_the_live_run_asks_for_exactly_one_strategy_through_the_shared_blend(
    tmp_path, fake_trainers, monkeypatch
):
    """#88: live is the one-strategy case of the call the harness makes with a whole sweep.

    Twice: once for the validation blend the gate reads, once for the live
    artifact that is submitted, and both through the same call (issue #97).
    """
    import zemir.blend as blend_module
    import zemir.pipeline as pipeline_module

    calls = []
    real = blend_module.blend_strategies

    def spy(strategies, *args, **kwargs):
        calls.append(dict(strategies))
        return real(strategies, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "blend_strategies", spy)
    strategy = _two_models(blend_neutralization=Neutralization(0.5, "medium"))

    _run(tmp_path, strategy, "shared")

    assert calls == [{"live": strategy}, {"live": strategy}]
