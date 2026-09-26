"""Numerai's own per-round scores for every submission slot: the only live scores.

Everything `zemir.scoring` computes is a backtest over validation, and a fixed
model scores the same eras identically every day, so none of it says what
Numerai scored or pays. This module asks Numerai instead, and records per
round exactly what it answers (see docs/adr/0003-live-scores-come-from-numerai.md).

A round's scores move daily until it resolves, so the record keeps one row per
(model, round), rewritten while the round is provisional and frozen once it has
resolved. Numerai revises provisional values; nothing here is a measurement of
ours worth keeping the history of.

The payout formula is not fixed: rounds up to 1342 pay `0.75·v2_corr20 +
2.25·mmc`, rounds from 1343 `3.0·corr60 + 15.0·mmc60`. So `payout_score` is
always computed from the multipliers Numerai lists on that round, never from a
constant. It is the score a round pays on, before stake, payout factor and
clipping turn it into NMR.

The two formulas' scores are on different scales, so no mean is taken over
`payout_score`. Every round is also priced by the current formula,
`REFERENCE_MULTIPLIERS`, as `reference_payout_score`: Numerai reports corr60
and mmc60 on the older rounds too, so the whole record is comparable on the
one formula that matters from here on, and that is what the headline averages.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from numerapi import NumerAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_SCORES_PATH = REPO_ROOT / "prod" / "zemir_01" / "live_scores.jsonl"

# The 20-day scores Numerai's model page shows as CORR20 and MMC20 in every era
# of the payout formula, so the headline reads the same numbers the site does.
CORR20 = "v2_corr20"
MMC20 = "mmc"
# The payout formula of rounds from 1343, as `round_model_performances_v2` lists
# it. Update it when Numerai's listed multipliers change again.
REFERENCE_MULTIPLIERS = {"corr60": 3.0, "mmc60": 15.0}


def _float(value) -> float | None:
    return None if value is None else float(value)


def _time(value) -> str | None:
    return None if value is None else value.isoformat() if hasattr(value, "isoformat") else str(value)


def fetch_rounds(model_id: str, *, napi: NumerAPI | None = None) -> list[dict]:
    """Numerai's per-round performance records for `model_id`, as returned.

    `round_model_performances_v2` is deprecated in favour of
    `submission_scores`, but only it carries each round's payout multipliers
    and whether the round has resolved; `submission_scores`' `resolved` is per
    daily score. Public data: no key needed.
    """
    napi = napi or NumerAPI()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return napi.round_model_performances_v2(model_id)


def _price(scores: Mapping[str, float | None], multipliers: Mapping[str, float]) -> float | None:
    """`Σ multiplier·score`, or None when there is no formula or a priced score is missing."""
    if not multipliers or any(scores.get(name) is None for name in multipliers):
        return None
    return sum(multiplier * scores[name] for name, multiplier in multipliers.items())


def round_row(model: str, model_id: str, performance: Mapping) -> dict:
    """One (model, round) row of the record, from one of `fetch_rounds`' records."""
    submission_scores = performance.get("submissionScores") or []
    scores = {s["displayName"]: _float(s["value"]) for s in submission_scores}
    percentiles = {s["displayName"]: _float(s["percentile"]) for s in submission_scores}
    days = [s["day"] for s in submission_scores if s.get("day") is not None]
    multipliers = {
        m["displayName"]: float(m["multiplier"]) for m in performance.get("payoutMultipliers") or []
    }
    return {
        "model": model,
        "model_id": model_id,
        "round": int(performance["roundNumber"]),
        "open_time": _time(performance.get("roundOpenTime")),
        "resolve_time": _time(performance.get("roundResolveTime")),
        "resolved": bool(performance.get("roundResolved")),
        "day": max(days) if days else None,
        "at_risk": _float(performance.get("atRisk")),
        "payout_factor": _float(performance.get("roundPayoutFactor")),
        "multipliers": multipliers,
        "payout_score": _price(scores, multipliers),
        "reference_payout_score": _price(scores, REFERENCE_MULTIPLIERS),
        "scores": scores,
        "percentiles": percentiles,
    }


def merge_rows(existing: Iterable[dict], fetched: Iterable[dict]) -> list[dict]:
    """The record after a fetch: fetched rows replace provisional ones, never resolved ones.

    A row already recorded as resolved is kept exactly as written. Rows Numerai
    no longer returns stay. Sorted by model, then round.
    """
    rows = {(row["model"], row["round"]): row for row in existing}
    for row in fetched:
        key = (row["model"], row["round"])
        if key in rows and rows[key]["resolved"]:
            continue
        rows[key] = row
    return [rows[key] for key in sorted(rows)]


def read_record(path: Path = LIVE_SCORES_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def update_record(
    models: Mapping[str, str],
    *,
    path: Path = LIVE_SCORES_PATH,
    napi: NumerAPI | None = None,
) -> list[dict]:
    """Fetch every slot in `models` (name to model id), merge, write, and return the record."""
    napi = napi or NumerAPI()
    fetched = [
        round_row(model, model_id, performance)
        for model, model_id in models.items()
        for performance in fetch_rounds(model_id, napi=napi)
    ]
    rows = merge_rows(read_record(path), fetched)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


@dataclass(frozen=True)
class ResolvedSummary:
    """Means over one model's resolved rounds. A mean over no rounds is None."""

    model: str
    resolved_rounds: int
    scored_rounds: int
    mean_corr20: float | None
    mean_mmc20: float | None
    mean_reference_payout_score: float | None


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize_resolved(rows: Iterable[dict]) -> list[ResolvedSummary]:
    """One summary per model, over resolved rounds only; provisional rounds are only counted."""
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    summaries = []
    for model, model_rows in sorted(by_model.items()):
        resolved = [row for row in model_rows if row["resolved"]]
        summaries.append(
            ResolvedSummary(
                model=model,
                resolved_rounds=len(resolved),
                scored_rounds=sum(1 for row in model_rows if row["scores"].get(CORR20) is not None),
                mean_corr20=_mean([row["scores"].get(CORR20) for row in resolved]),
                mean_mmc20=_mean([row["scores"].get(MMC20) for row in resolved]),
                mean_reference_payout_score=_mean([row["reference_payout_score"] for row in resolved]),
            )
        )
    return summaries


_REFERENCE_LABEL = " + ".join(f"{m:g}*{name}" for name, m in REFERENCE_MULTIPLIERS.items())


def format_resolved(summaries: Iterable[ResolvedSummary]) -> str:
    """The live headline: Numerai's scores, resolved rounds only."""

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    lines = ["live (Numerai's scores, resolved rounds only):"]
    for s in summaries:
        lines.append(
            f"  {s.model}: {s.resolved_rounds} resolved of {s.scored_rounds} scored rounds  "
            f"corr20={number(s.mean_corr20)}  mmc20={number(s.mean_mmc20)}  "
            f"payout_score={number(s.mean_reference_payout_score)} ({_REFERENCE_LABEL})"
        )
    return "\n".join(lines)
