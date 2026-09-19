import pandas as pd
import pytest

from zemir.config import LIVE, MIN_VALIDATION_MEAN_CORR
from zemir.pipeline import combine_predictions, run_pipeline

from factories import SIGNAL_COLUMN, make_dataset, predict_column_trainer, predict_negated_column_trainer


def test_combine_predictions_returns_a_lone_model_untouched():
    preds = pd.Series([0.1, 0.9, 0.3])
    era = pd.Series(["0001", "0001", "0001"])

    combined = combine_predictions({"linear": preds}, era)

    pd.testing.assert_series_equal(combined, preds.rename("prediction"))


def test_combine_predictions_blends_per_era_ranks_by_weight():
    # One era, two models. A=[10,20,30,40] ranks to [.25,.5,.75,1.0]; B is A
    # reversed and so ranks to [1.0,.75,.5,.25]. At weights 0.75/0.25 the
    # blend is 0.75*A_rank + 0.25*B_rank.
    era = pd.Series(["0001"] * 4)
    a = pd.Series([10.0, 20.0, 30.0, 40.0])
    b = pd.Series([40.0, 30.0, 20.0, 10.0])

    combined = combine_predictions({"a": a, "b": b}, era, {"a": 0.75, "b": 0.25})

    assert combined.tolist() == pytest.approx([0.4375, 0.5625, 0.6875, 0.8125])


def test_combine_predictions_ranks_within_each_era_independently():
    # Same relative pattern in both eras, but at very different magnitudes —
    # a rank blend should produce the identical result in each, since it
    # never compares magnitudes across eras.
    era = pd.Series(["0001", "0001", "0002", "0002"])
    a = pd.Series([1.0, 2.0, 100.0, 200.0])
    b = pd.Series([2.0, 1.0, 200.0, 100.0])

    combined = combine_predictions({"a": a, "b": b}, era)

    assert combined.tolist() == pytest.approx([0.75, 0.75, 0.75, 0.75])


def test_run_pipeline_exercises_the_post_fit_path_from_a_synthetic_dataset(tmp_path):
    """Accepting a `Dataset` argument is what makes this run in milliseconds (issue #75)."""
    dataset = make_dataset()

    result = run_pipeline(
        LIVE,
        run_id="test-run",
        trainers={"linear": predict_column_trainer(SIGNAL_COLUMN)},
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


def test_run_pipeline_gate_passes_when_predictions_track_the_target(tmp_path):
    dataset = make_dataset()

    result = run_pipeline(
        LIVE,
        run_id="above-gate",
        trainers={"linear": predict_column_trainer(SIGNAL_COLUMN)},
        dataset=dataset,
        runs_dir=tmp_path,
    )

    # scripts/run_pipeline.py's gate: `mean_corr < MIN_VALIDATION_MEAN_CORR` -> don't submit.
    assert result.combined_validation_score.mean_corr >= MIN_VALIDATION_MEAN_CORR


def test_run_pipeline_gate_fails_when_predictions_are_anti_correlated_with_the_target(tmp_path):
    dataset = make_dataset()

    result = run_pipeline(
        LIVE,
        run_id="below-gate",
        trainers={"linear": predict_negated_column_trainer(SIGNAL_COLUMN)},
        dataset=dataset,
        runs_dir=tmp_path,
    )

    assert result.combined_validation_score.mean_corr < MIN_VALIDATION_MEAN_CORR
