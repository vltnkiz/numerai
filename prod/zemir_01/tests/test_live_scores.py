"""Numerai's per-round record (docs/adr/0003), from canned API responses: nothing here reaches Numerai."""

import json
from datetime import datetime, timezone

import pytest

from zemir.live_scores import (
    format_resolved,
    merge_rows,
    read_record,
    round_row,
    summarize_resolved,
    update_record,
)

# The two payout formulas `round_model_performances_v2` has listed for zemir_01:
# rounds up to 1342, and rounds from 1343.
_OLD_FORMULA = [{"displayName": "v2_corr20", "multiplier": 0.75}, {"displayName": "mmc", "multiplier": 2.25}]
_NEW_FORMULA = [{"displayName": "corr60", "multiplier": 3.0}, {"displayName": "mmc60", "multiplier": 15.0}]


def _performance(number, *, resolved, multipliers, scores, day=20):
    return {
        "roundNumber": number,
        "roundOpenTime": datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
        "roundResolveTime": datetime(2026, 9, 25, 16, tzinfo=timezone.utc),
        "roundResolved": resolved,
        "atRisk": "0E-18",
        "roundPayoutFactor": "1.000000000000000000",
        "payoutMultipliers": multipliers,
        "submissionScores": [
            {"displayName": name, "value": value, "percentile": 0.5, "day": day if value is not None else None}
            for name, value in scores.items()
        ],
    }


class _FakeNumerAPI:
    def __init__(self, by_model_id):
        self.by_model_id = by_model_id

    def round_model_performances_v2(self, model_id):
        return self.by_model_id[model_id]


def test_payout_score_uses_the_rounds_own_multipliers():
    old = round_row("m", "id", _performance(
        1340, resolved=True, multipliers=_OLD_FORMULA,
        scores={"v2_corr20": 0.004, "mmc": -0.002, "corr60": 0.009, "mmc60": 0.001},
    ))
    new = round_row("m", "id", _performance(
        1343, resolved=False, multipliers=_NEW_FORMULA,
        scores={"v2_corr20": 0.004, "mmc": -0.002, "corr60": 0.009, "mmc60": 0.001},
    ))

    assert old["payout_score"] == pytest.approx(0.75 * 0.004 + 2.25 * -0.002)
    assert new["payout_score"] == pytest.approx(3.0 * 0.009 + 15.0 * 0.001)


def test_every_round_is_also_priced_by_the_current_formula():
    old = round_row("m", "id", _performance(
        1340, resolved=True, multipliers=_OLD_FORMULA,
        scores={"v2_corr20": 0.004, "mmc": -0.002, "corr60": 0.009, "mmc60": 0.001},
    ))

    assert old["reference_payout_score"] == pytest.approx(3.0 * 0.009 + 15.0 * 0.001)


def test_a_round_missing_a_priced_score_has_no_payout_score():
    row = round_row("m", "id", _performance(
        1364, resolved=False, multipliers=_NEW_FORMULA, scores={"corr60": None, "mmc60": None},
    ))

    assert row["payout_score"] is None and row["reference_payout_score"] is None
    assert row["day"] is None


def test_a_row_records_what_numerai_returned_as_json_plain_values():
    row = round_row("zemir_01", "id-1", _performance(
        1340, resolved=True, multipliers=_OLD_FORMULA, scores={"v2_corr20": 0.004, "mmc": -0.002},
    ))

    assert row == json.loads(json.dumps(row))
    assert row["model"] == "zemir_01" and row["round"] == 1340 and row["resolved"] is True
    assert row["at_risk"] == 0.0 and row["payout_factor"] == 1.0
    assert row["multipliers"] == {"v2_corr20": 0.75, "mmc": 2.25}
    assert row["scores"] == {"v2_corr20": 0.004, "mmc": -0.002}
    assert row["day"] == 20


def test_a_provisional_row_is_replaced_and_a_resolved_one_is_frozen():
    frozen = {"model": "m", "round": 1, "resolved": True, "payout_score": 0.1}
    provisional = {"model": "m", "round": 2, "resolved": False, "payout_score": 0.2}
    gone = {"model": "m", "round": 0, "resolved": False, "payout_score": 0.0}

    merged = merge_rows(
        [provisional, frozen, gone],
        [
            {"model": "m", "round": 1, "resolved": True, "payout_score": 9.9},
            {"model": "m", "round": 2, "resolved": True, "payout_score": 0.3},
        ],
    )

    assert [(r["round"], r["payout_score"]) for r in merged] == [(0, 0.0), (1, 0.1), (2, 0.3)]


def test_update_record_fetches_every_slot_and_rewrites_the_file(tmp_path):
    path = tmp_path / "live_scores.jsonl"
    napi = _FakeNumerAPI({
        "id-a": [_performance(1340, resolved=True, multipliers=_OLD_FORMULA, scores={"v2_corr20": 0.01, "mmc": 0.0})],
        "id-b": [_performance(1341, resolved=False, multipliers=_OLD_FORMULA, scores={"v2_corr20": 0.02, "mmc": 0.0})],
    })

    rows = update_record({"a": "id-a", "b": "id-b"}, path=path, napi=napi)
    update_record({"a": "id-a", "b": "id-b"}, path=path, napi=napi)

    assert [(r["model"], r["round"]) for r in rows] == [("a", 1340), ("b", 1341)]
    assert read_record(path) == rows


def test_the_headline_averages_resolved_rounds_only_on_the_current_formula():
    rows = [
        round_row("m", "id", _performance(1339, resolved=True, multipliers=_OLD_FORMULA, scores={"v2_corr20": 0.002, "mmc": -0.001, "corr60": 0.004, "mmc60": -0.001})),
        round_row("m", "id", _performance(1343, resolved=True, multipliers=_NEW_FORMULA, scores={"v2_corr20": 0.004, "mmc": -0.003, "corr60": 0.010, "mmc60": 0.001})),
        round_row("m", "id", _performance(1344, resolved=False, multipliers=_NEW_FORMULA, scores={"v2_corr20": 0.03, "mmc": 0.02, "corr60": 0.03, "mmc60": 0.02})),
        round_row("m", "id", _performance(1364, resolved=False, multipliers=_NEW_FORMULA, scores={"v2_corr20": None})),
    ]

    [summary] = summarize_resolved(rows)

    assert (summary.resolved_rounds, summary.scored_rounds) == (2, 3)
    assert summary.mean_corr20 == pytest.approx(0.003)
    assert summary.mean_mmc20 == pytest.approx(-0.002)
    assert summary.mean_reference_payout_score == pytest.approx(3.0 * 0.007 + 15.0 * 0.0)
    headline = format_resolved([summary])
    assert "m: 2 resolved of 3 scored rounds" in headline
    assert "(3*corr60 + 15*mmc60)" in headline


def test_a_model_with_nothing_resolved_says_so_rather_than_printing_zero():
    rows = [round_row("m", "id", _performance(1343, resolved=False, multipliers=_NEW_FORMULA, scores={"v2_corr20": 0.03}))]

    [summary] = summarize_resolved(rows)

    assert summary.mean_corr20 is None
    assert "corr20=n/a" in format_resolved([summary])
