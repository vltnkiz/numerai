"""Whether a scheduled live run has anything to do — decided by round state, not the calendar.

Issue #69: the live run is started by Windows Task Scheduler from several
triggers (a daily 12:00 UTC trigger that also fires late after a missed start,
plus a logon trigger), so any given invocation may be early, on time, late, or
a duplicate. Rather than each trigger knowing what it is for, every invocation
asks the same question — is there an opened round this machine hasn't finished
yet? — and only the nominal-open window is allowed to wait for one.

"Finished" is a local marker per round number under `runs/rounds/`, written
only for outcomes a retry cannot change: a submission, or a validation-gate
failure (training is deterministic, so a rerun fails identically). Anything
else — a crash, a download error, the free-memory guard — leaves no marker, and
the next trigger tries again.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from numerapi import NumerAPI

from zemir.config import ROUND_OPEN_MAX_WAIT_SECONDS, ROUND_OPEN_POLL_INTERVAL_SECONDS

ROUND_MARKERS_DIR = Path(__file__).resolve().parents[1] / "runs" / "rounds"

# Numerai Classic opens a round Tuesday-Saturday at a nominal 12:00 UTC
# (`datetime.weekday()`: Monday is 0). Only used to bound *waiting* — whether
# a round exists at all is always read from Numerai, never inferred from this.
ROUND_OPEN_WEEKDAYS = frozenset({1, 2, 3, 4, 5})
NOMINAL_ROUND_OPEN_HOUR_UTC = 12
# A trigger scheduled for 12:00:00 can fire a moment early; don't let that
# turn the one invocation that is supposed to wait into one that exits.
_WAIT_WINDOW_EARLY_SLACK = timedelta(minutes=5)

SUBMITTED = "submitted"
GATE_FAILED = "gate_failed"
_FINAL_OUTCOMES = frozenset({SUBMITTED, GATE_FAILED})

# `rounds(number: 0)` is Numerai's alias for the current round. Not
# `NumerAPI.check_round_open()`: that is `openTime < now < closeStakingTime`,
# i.e. "still stakeable", while a late submission is still accepted and scored
# (unstaked) after staking closes, and issue #69 decided to submit those.
_CURRENT_ROUND_QUERY = """
    query($tournament: Int!) {
      rounds(tournament: $tournament, number: 0) {
        number
        openTime
        closeStakingTime
      }
    }
"""


class RoundNotOpen(RuntimeError):
    pass


class InsufficientMemory(RuntimeError):
    pass


@dataclass(frozen=True)
class Round:
    number: int
    open_time: datetime
    close_staking_time: datetime

    def is_late(self, now: datetime) -> bool:
        """True once staking has closed: an upload is still scored, but cannot be staked."""
        return now >= self.close_staking_time


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fetch_current_round(napi: NumerAPI) -> Round | None:
    """Numerai's current round, or None in the between-rounds gaps where it has none."""
    try:
        raw = napi.raw_query(_CURRENT_ROUND_QUERY, {"tournament": napi.tournament_id})
    except ValueError:
        # numerapi raises ValueError for GraphQL errors, which is how some
        # between-rounds periods answer `number: 0` (see check_round_open).
        return None
    rounds = raw["data"]["rounds"]
    if not rounds or rounds[0] is None:
        return None
    return Round(
        number=int(rounds[0]["number"]),
        open_time=datetime.fromisoformat(rounds[0]["openTime"]),
        close_staking_time=datetime.fromisoformat(rounds[0]["closeStakingTime"]),
    )


def record_round_outcome(
    round_number: int, outcome: str, *, run_id: str, markers_dir: Path = ROUND_MARKERS_DIR
) -> None:
    """Mark `round_number` finished, so later triggers for the same round exit cleanly."""
    if outcome not in _FINAL_OUTCOMES:
        raise ValueError(f"not a final outcome: {outcome!r} (have: {sorted(_FINAL_OUTCOMES)})")
    markers_dir.mkdir(parents=True, exist_ok=True)
    (markers_dir / f"{round_number}.json").write_text(
        json.dumps({"round": round_number, "outcome": outcome, "run_id": run_id}, indent=2)
    )


def is_round_done(round_number: int, *, markers_dir: Path = ROUND_MARKERS_DIR) -> bool:
    return (markers_dir / f"{round_number}.json").exists()


def wait_deadline(now: datetime) -> datetime | None:
    """End of today's wait window if `now` is inside it, else None.

    The window is the nominal open plus `ROUND_OPEN_MAX_WAIT_SECONDS`, on round
    days only. Outside it an invocation never waits: a logon at 08:00 or a
    Sunday catch-up checks once and exits, rather than sleeping for hours in
    front of a round that opens later — the 12:00 UTC trigger handles that.
    """
    if now.weekday() not in ROUND_OPEN_WEEKDAYS:
        return None
    nominal_open = now.replace(
        hour=NOMINAL_ROUND_OPEN_HOUR_UTC, minute=0, second=0, microsecond=0
    )
    deadline = nominal_open + timedelta(seconds=ROUND_OPEN_MAX_WAIT_SECONDS)
    if nominal_open - _WAIT_WINDOW_EARLY_SLACK <= now < deadline:
        return deadline
    return None


def round_to_run(
    *,
    napi: NumerAPI,
    markers_dir: Path = ROUND_MARKERS_DIR,
    poll_interval_seconds: int = ROUND_OPEN_POLL_INTERVAL_SECONDS,
    clock: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
) -> Round | None:
    """The opened, unfinished round this invocation should run for — or None if there is none.

    Late rounds count: a round whose staking has closed but that this machine
    never finished is still returned (issue #69 submits late rather than skip).
    Inside the nominal-open wait window, polls until an unfinished round opens,
    raising `RoundNotOpen` at the window's end — issue #33's fail-loud posture,
    since that is the one time a round is expected and missing.

    Inside the window only today's round counts. An unfinished round from an
    earlier day is left for a later trigger: fitting it now would see today's
    round open mid-fit, throw the fit away, and — since the task never starts a
    second instance — leave nothing running to catch today's round.
    """
    deadline = wait_deadline(clock())
    earliest_open = (
        None
        if deadline is None
        else deadline - timedelta(seconds=ROUND_OPEN_MAX_WAIT_SECONDS) - _WAIT_WINDOW_EARLY_SLACK
    )
    while True:
        current = fetch_current_round(napi)
        now = clock()
        stale = (
            current is not None and earliest_open is not None and current.open_time < earliest_open
        )
        if (
            current is not None
            and current.open_time <= now
            and not stale
            and not is_round_done(current.number, markers_dir=markers_dir)
        ):
            return current
        if deadline is None:
            return None
        if now >= deadline:
            if current is None:
                latest = "no current round"
            elif is_round_done(current.number, markers_dir=markers_dir):
                latest = f"round {current.number} already finished"
            else:
                latest = f"round {current.number} opened before today"
            raise RoundNotOpen(
                f"no unfinished round opened by {deadline.isoformat()} ({latest})"
            )
        sleep(poll_interval_seconds)


def available_memory_bytes() -> int:
    """Physical memory the OS could hand out right now (Windows "available", not just free)."""
    if sys.platform == "win32":

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return status.ullAvailPhys
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def require_available_memory(
    min_gib: float, *, available: Callable[[], int] = available_memory_bytes
) -> None:
    """Refuse to start a fit the machine can't hold, rather than page or die partway through."""
    have_gib = available() / 2**30
    if have_gib < min_gib:
        raise InsufficientMemory(
            f"{have_gib:.1f} GiB available < {min_gib:g} GiB required to start the fit"
        )
