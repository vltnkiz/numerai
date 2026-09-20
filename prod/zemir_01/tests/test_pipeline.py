import json
from dataclasses import replace

import pandas as pd
import pytest

from zemir.config import LIVE, MIN_VALIDATION_MEAN_CORR
from zemir.pipeline import run_pipeline
from zemir.strategy import BlendSpec, Neutralization, Strategy

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
    assert "linear" in result.validation_scores
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

    # scripts/run_pipeline.py's gate: `mean_corr < MIN_VALIDATION_MEAN_CORR` -> don't submit.
    assert result.combined_validation_score.mean_corr >= MIN_VALIDATION_MEAN_CORR


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

    assert result.combined_validation_score.mean_corr < MIN_VALIDATION_MEAN_CORR


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

    assert set(result.validation_scores) == {"narrow_model", "wide_model"}
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
    """#88: live is the one-strategy case of the call the harness makes with a whole sweep."""
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

    assert calls == [{"live": strategy}]
