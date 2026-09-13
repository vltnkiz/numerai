from datetime import datetime, timedelta, timezone

import pytest

from zemir.schedule import (
    GATE_FAILED,
    SUBMITTED,
    InsufficientMemory,
    RoundNotOpen,
    fetch_current_round,
    is_round_done,
    record_round_outcome,
    require_available_memory,
    round_to_run,
)

UTC = timezone.utc
# 2026-09-15 is a Tuesday, 2026-09-13 a Sunday.
TUESDAY_NOON = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
SUNDAY_MORNING = datetime(2026, 9, 13, 9, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime):
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


class FakeNumerAPI:
    """Answers `rounds(number: 0)` from a function of the fake clock's time."""

    tournament_id = 8

    def __init__(self, clock: FakeClock, rounds_at):
        self.clock = clock
        self.rounds_at = rounds_at

    def raw_query(self, query, variables):
        current = self.rounds_at(self.clock.now)
        if isinstance(current, Exception):
            raise current
        if current is None:
            return {"data": {"rounds": [None]}}
        number, open_time, close_staking = current
        return {
            "data": {
                "rounds": [
                    {
                        "number": number,
                        "openTime": open_time.isoformat().replace("+00:00", "Z"),
                        "closeStakingTime": close_staking.isoformat().replace("+00:00", "Z"),
                    }
                ]
            }
        }


def saturday_round():
    opened = datetime(2026, 9, 12, 12, 4, 8, tzinfo=UTC)
    return (1354, opened, datetime(2026, 9, 13, 16, 0, tzinfo=UTC))


def tuesday_round(opens_at: datetime):
    return (1355, opens_at, opens_at + timedelta(hours=1))


def run(clock, rounds_at, markers_dir):
    return round_to_run(
        napi=FakeNumerAPI(clock, rounds_at),
        markers_dir=markers_dir,
        poll_interval_seconds=120,
        clock=clock,
        sleep=clock.sleep,
    )


def test_unfinished_open_round_runs_immediately(tmp_path):
    clock = FakeClock(SUNDAY_MORNING)
    found = run(clock, lambda now: saturday_round(), tmp_path)
    assert found.number == 1354
    assert clock.sleeps == []


def test_finished_round_outside_wait_window_exits_without_waiting(tmp_path):
    record_round_outcome(1354, SUBMITTED, run_id="r", markers_dir=tmp_path)
    clock = FakeClock(SUNDAY_MORNING)
    assert run(clock, lambda now: saturday_round(), tmp_path) is None
    assert clock.sleeps == []


def test_early_logon_on_a_round_day_does_not_wait_for_noon(tmp_path):
    record_round_outcome(1354, SUBMITTED, run_id="r", markers_dir=tmp_path)
    clock = FakeClock(TUESDAY_NOON.replace(hour=8))
    assert run(clock, lambda now: saturday_round(), tmp_path) is None
    assert clock.sleeps == []


def test_gate_failure_counts_as_finished(tmp_path):
    record_round_outcome(1354, GATE_FAILED, run_id="r", markers_dir=tmp_path)
    assert run(FakeClock(SUNDAY_MORNING), lambda now: saturday_round(), tmp_path) is None


def test_noon_trigger_waits_for_the_new_round_to_open(tmp_path):
    record_round_outcome(1354, SUBMITTED, run_id="r", markers_dir=tmp_path)
    opens_at = TUESDAY_NOON + timedelta(minutes=5)

    def rounds_at(now):
        return tuesday_round(opens_at) if now >= opens_at else saturday_round()

    clock = FakeClock(TUESDAY_NOON)
    found = run(clock, rounds_at, tmp_path)
    assert found.number == 1355
    assert clock.sleeps == [120, 120, 120]


def test_noon_trigger_fails_loudly_when_no_round_opens_in_the_window(tmp_path):
    record_round_outcome(1354, SUBMITTED, run_id="r", markers_dir=tmp_path)
    clock = FakeClock(TUESDAY_NOON)
    with pytest.raises(RoundNotOpen, match="round 1354 already finished"):
        run(clock, lambda now: saturday_round(), tmp_path)
    assert clock.now >= TUESDAY_NOON + timedelta(minutes=45)


def test_noon_trigger_waits_for_todays_round_over_an_unfinished_stale_one(tmp_path):
    # Saturday's round was never finished (machine off since). Fitting it at
    # noon would see today's round open mid-fit and throw the fit away.
    opens_at = TUESDAY_NOON + timedelta(minutes=5)

    def rounds_at(now):
        return tuesday_round(opens_at) if now >= opens_at else saturday_round()

    clock = FakeClock(TUESDAY_NOON)
    found = run(clock, rounds_at, tmp_path)
    assert found.number == 1355
    assert clock.sleeps == [120, 120, 120]


def test_noon_trigger_fails_loudly_when_only_a_stale_round_is_open(tmp_path):
    clock = FakeClock(TUESDAY_NOON)
    with pytest.raises(RoundNotOpen, match="round 1354 opened before today"):
        run(clock, lambda now: saturday_round(), tmp_path)


def test_between_rounds_gap_keeps_polling_inside_the_window(tmp_path):
    opens_at = TUESDAY_NOON + timedelta(minutes=3)

    def rounds_at(now):
        return tuesday_round(opens_at) if now >= opens_at else ValueError("not open")

    found = run(FakeClock(TUESDAY_NOON), rounds_at, tmp_path)
    assert found.number == 1355


def test_between_rounds_gap_outside_the_window_is_nothing_to_do(tmp_path):
    assert run(FakeClock(SUNDAY_MORNING), lambda now: None, tmp_path) is None


def test_late_round_is_still_returned_and_reported_late(tmp_path):
    clock = FakeClock(datetime(2026, 9, 14, 9, 0, tzinfo=UTC))  # Monday, after Sunday 16:00 close
    found = run(clock, lambda now: saturday_round(), tmp_path)
    assert found.number == 1354
    assert found.is_late(clock.now)
    assert not found.is_late(SUNDAY_MORNING)


def test_fetch_current_round_parses_numerai_timestamps():
    clock = FakeClock(SUNDAY_MORNING)
    current = fetch_current_round(FakeNumerAPI(clock, lambda now: saturday_round()))
    assert current.number == 1354
    assert current.close_staking_time == datetime(2026, 9, 13, 16, 0, tzinfo=UTC)


def test_markers_only_accept_final_outcomes(tmp_path):
    with pytest.raises(ValueError):
        record_round_outcome(1354, "crashed", run_id="r", markers_dir=tmp_path)
    assert not is_round_done(1354, markers_dir=tmp_path)


def test_memory_guard(tmp_path):
    require_available_memory(46, available=lambda: 50 * 2**30)
    with pytest.raises(InsufficientMemory, match="45.0 GiB available < 46 GiB"):
        require_available_memory(46, available=lambda: 45 * 2**30)
