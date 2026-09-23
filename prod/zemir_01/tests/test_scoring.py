from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zemir.scoring import (
    PredictionSanityError,
    era_max_feature_corr,
    era_mmc,
    era_numerai_corr,
    era_spearman,
    rank_normalize,
    score_predictions,
    summarize_era_scores,
    summarize_era_spearman,
    validate_predictions,
)

GOLDENS = Path(__file__).parent / "data"


def test_rank_normalize_maps_into_the_open_unit_interval_preserving_order():
    values = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0], index=["a", "b", "c", "d", "e"])

    normalized = rank_normalize(values)

    assert normalized.min() > 0.0
    assert normalized.max() < 1.0
    assert list(normalized.sort_values().index) == list(values.sort_values().index)


def test_validate_predictions_accepts_well_formed_predictions():
    predictions = pd.Series([0.1, 0.5, 0.9, 0.3])
    era = pd.Series(["0001", "0001", "0002", "0002"])

    validate_predictions(predictions, era)  # must not raise


def test_validate_predictions_rejects_nulls():
    predictions = pd.Series([0.1, None, 0.9])
    era = pd.Series(["0001", "0001", "0001"])

    with pytest.raises(PredictionSanityError, match="null prediction"):
        validate_predictions(predictions, era)


def test_validate_predictions_rejects_values_outside_open_unit_interval():
    predictions = pd.Series([0.1, 0.5, 1.0])
    era = pd.Series(["0001", "0001", "0001"])

    with pytest.raises(PredictionSanityError, match=r"outside \(0, 1\)"):
        validate_predictions(predictions, era)


def test_validate_predictions_rejects_a_zero_variance_era():
    predictions = pd.Series([0.5, 0.5, 0.5, 0.2, 0.8])
    era = pd.Series(["0001", "0001", "0001", "0002", "0002"])

    with pytest.raises(PredictionSanityError, match=r"zero-variance predictions in era\(s\) \['0001'\]"):
        validate_predictions(predictions, era)


# ---------------------------------------------------------------------------
# Issue #96: `ValidationScore` and `score_validation` were deleted, and their
# arithmetic moved to `summarize_era_spearman`. These two tests exist to prove
# the live run's numbers did not move when the plumbing did. A failure here
# means a recorded `score_log.jsonl` series has silently changed meaning.
# ---------------------------------------------------------------------------


def _synthetic_era_spearman() -> pd.Series:
    """A deterministic 40-era Spearman series. PCG64 is stable across numpy versions."""
    rng = np.random.default_rng(0)
    eras = np.repeat([f"{i:04d}" for i in range(40)], 50)
    target = rng.normal(size=2000)
    prediction = target * 0.05 + rng.normal(size=2000)
    return era_spearman(pd.DataFrame({"era": eras, "target": target, "prediction": prediction}))


def test_summarize_era_spearman_holds_the_retired_score_validation_arithmetic():
    # Literals captured from `score_validation` on main @ 7504cbb, before the
    # deletion. Exact equality is the point: this pins the arithmetic, not an
    # approximation of it.
    score = summarize_era_spearman(_synthetic_era_spearman())

    assert score.mean_corr == 0.025743097238895556
    assert score.std_corr == 0.1387847745216482
    assert score.sharpe == 0.18548934728340857
    assert score.smart_sharpe == 0.19109662534983185


def test_summarize_era_spearman_reproduces_a_recorded_production_score_log_entry():
    """Replays the real 655-era series from run 20260923T120411Z.

    Its `combined` numbers are committed in `score_log.jsonl`, so this checks
    the refactor against production output rather than a fixture. `rel=1e-12`
    covers the CSV round-trip, which is the only source of drift: the series
    is re-read from text rather than recomputed.
    """
    corrs = pd.read_csv(
        GOLDENS / "validation_era_corr_20260923T120411Z.csv", dtype={"era": str}
    ).set_index("era")["corr"]

    score = summarize_era_spearman(corrs)

    assert len(corrs) == 655
    assert score.mean_corr == pytest.approx(0.016447134682809162, rel=1e-12)
    assert score.std_corr == pytest.approx(0.015879705826150687, rel=1e-12)
    assert score.sharpe == pytest.approx(1.0357329577053016, rel=1e-12)
    assert score.smart_sharpe == pytest.approx(0.41590757858848304, rel=1e-12)


def test_the_two_vocabularies_report_different_numbers_under_the_same_names():
    """Spearman `mean_corr` and paid `mean_corr` are different quantities.

    The whole reason `era_spearman` keeps a name of its own. If this ever
    starts passing by equality, one of the two has stopped being what it says.
    """
    frame = _paid_fixture()
    one = frame["predictions"][["a"]]

    spearman = summarize_era_spearman(
        era_spearman(
            pd.DataFrame(
                {
                    "era": frame["era"],
                    "target": frame["target"],
                    "prediction": one["a"],
                }
            )
        )
    )
    paid = summarize_era_scores(
        era_numerai_corr(one, frame["target"], frame["era"]),
        era_mmc(one, frame["target"], frame["era"], frame["meta_model"]),
        era_max_feature_corr(one, frame["features"], frame["era"]),
    )

    assert spearman.mean_corr != paid.loc["a", "mean_corr"]


def _paid_fixture() -> dict:
    """Small aligned frames for the paid vocabulary: two prediction columns, 12 eras."""
    rng = np.random.default_rng(7)
    n_eras, per_era = 12, 60
    era = pd.Series(np.repeat([f"{i:04d}" for i in range(n_eras)], per_era), name="era")
    n = len(era)
    target = pd.Series(rng.normal(size=n), name="target")
    predictions = pd.DataFrame(
        {
            "a": target * 0.10 + rng.normal(size=n),
            "b": target * 0.02 + rng.normal(size=n),
        }
    )
    features = pd.DataFrame(
        {f"feature_{i}": rng.normal(size=n) for i in range(5)},
    )
    # The meta model covers only the back half of the eras, as it does in
    # production: MMC is measured over a window, never all of validation.
    covered = era.isin([f"{i:04d}" for i in range(n_eras // 2, n_eras)])
    meta_model = pd.Series(rng.normal(size=int(covered.sum())), index=era.index[covered])
    return {
        "era": era,
        "target": target,
        "predictions": predictions,
        "features": features,
        "meta_model": meta_model,
    }


def test_score_predictions_composes_the_primitives_it_replaced():
    """The shared scorer is exactly the four calls `score_configs` used to inline."""
    f = _paid_fixture()

    scores = score_predictions(
        f["predictions"],
        f["target"],
        f["era"],
        meta_model=f["meta_model"],
        features=f["features"],
    )

    pd.testing.assert_frame_equal(
        scores.corr_by_era, era_numerai_corr(f["predictions"], f["target"], f["era"])
    )
    pd.testing.assert_frame_equal(
        scores.mmc_by_era,
        era_mmc(f["predictions"], f["target"], f["era"], f["meta_model"]),
    )
    pd.testing.assert_frame_equal(
        scores.exposure_by_era,
        era_max_feature_corr(f["predictions"], f["features"], f["era"]),
    )
    pd.testing.assert_frame_equal(
        scores.summary,
        summarize_era_scores(scores.corr_by_era, scores.mmc_by_era, scores.exposure_by_era),
    )


def test_score_predictions_measures_mmc_over_the_window_and_corr_over_everything():
    """The meta-model window is narrower than validation, and the summary says so."""
    f = _paid_fixture()

    scores = score_predictions(
        f["predictions"],
        f["target"],
        f["era"],
        meta_model=f["meta_model"],
        features=f["features"],
    )

    assert len(scores.corr_by_era) == 12
    assert len(scores.mmc_by_era) == 6
    assert list(scores.summary["eras"]) == [12, 12]
    assert list(scores.summary["mmc_eras"]) == [6, 6]
    assert list(scores.summary.index) == ["a", "b"]


def test_score_predictions_writes_nothing(tmp_path, monkeypatch):
    """Pure by design: artifact layout belongs to the caller, not the shared scorer."""
    f = _paid_fixture()
    monkeypatch.chdir(tmp_path)

    score_predictions(
        f["predictions"],
        f["target"],
        f["era"],
        meta_model=f["meta_model"],
        features=f["features"],
    )

    assert list(tmp_path.iterdir()) == []
