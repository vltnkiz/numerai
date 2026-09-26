import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from zemir import data
from zemir.data import load_meta_model, refresh_validation


class FakeNumerAPI:
    """Serves one meta model frame, or fails every download when `frame` is None."""

    def __init__(self, frame: pd.DataFrame | None):
        self.frame = frame
        self.downloads: list[str] = []

    def download_dataset(self, filename: str, dest_path: str) -> None:
        self.downloads.append(filename)
        if self.frame is None:
            raise ConnectionError("no network")
        self.frame.to_parquet(dest_path)


def _meta_model(eras: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "era": [f"{era:04d}" for era in eras],
            "data_type": "validation",
            "numerai_meta_model": [0.5] * len(eras),
        },
        index=pd.Index([f"id{era}" for era in eras], name="id"),
    )


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATASETS_DIR", tmp_path)
    return tmp_path


def _cache(datasets_dir, frame: pd.DataFrame) -> None:
    (datasets_dir / "5.3").mkdir()
    frame.to_parquet(datasets_dir / "5.3" / "meta_model.parquet")


def test_load_meta_model_reads_the_cached_copy_without_touching_the_network(datasets_dir):
    _cache(datasets_dir, _meta_model(range(1133, 1222)))
    napi = FakeNumerAPI(_meta_model(range(1133, 1225)))

    meta_model = load_meta_model("5.3", napi=napi)

    assert napi.downloads == []
    assert len(meta_model) == 89


def test_refresh_replaces_a_stale_cached_copy(datasets_dir):
    """The live path's refresh lengthens the window every later reader sees (issue #101)."""
    _cache(datasets_dir, _meta_model(range(1133, 1222)))
    napi = FakeNumerAPI(_meta_model(range(1133, 1225)))

    assert len(load_meta_model("5.3", napi=napi, refresh=True)) == 92
    assert len(load_meta_model("5.3", napi=FakeNumerAPI(None))) == 92


def test_a_failed_refresh_falls_back_to_the_intact_cached_copy(datasets_dir):
    """A stale window is correct numbers over fewer eras — better than no MMC at all."""
    _cache(datasets_dir, _meta_model(range(1133, 1222)))

    with pytest.warns(RuntimeWarning, match="stale"):
        meta_model = load_meta_model("5.3", napi=FakeNumerAPI(None), refresh=True)

    assert len(meta_model) == 89


def test_a_failed_refresh_with_nothing_cached_raises(datasets_dir):
    with pytest.raises(ConnectionError):
        load_meta_model("5.3", napi=FakeNumerAPI(None), refresh=True)


# `refresh_validation`: the live run's post-upload refresh of validation.parquet.


def _validation(eras: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "era": [f"{era:04d}" for era in eras],
            "data_type": "validation",
            "feature_a": [0.5] * len(eras),
            "target": [0.5] * len(eras),
        },
        index=pd.Index([f"id{era}" for era in eras], name="id"),
    )


def _cache_validation(datasets_dir, frame: pd.DataFrame, *, age_days: float) -> Path:
    (datasets_dir / "5.3").mkdir(exist_ok=True)
    path = datasets_dir / "5.3" / "validation.parquet"
    frame.to_parquet(path)
    stamp = (NOW - timedelta(days=age_days)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _eras_on_disk(path: Path) -> int:
    return pd.read_parquet(path, columns=["era"])["era"].nunique()


def _leftovers(datasets_dir) -> list[str]:
    return sorted(p.name for p in (datasets_dir / "5.3").iterdir() if p.name != "validation.parquet")


class GarbageNumerAPI(FakeNumerAPI):
    """Serves bytes that are not a parquet file, as a truncated download would."""

    def download_dataset(self, filename: str, dest_path: str) -> None:
        self.downloads.append(filename)
        Path(dest_path).write_bytes(b"not a parquet file")


NOW = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


def test_a_copy_younger_than_a_week_is_left_alone(datasets_dir):
    path = _cache_validation(datasets_dir, _validation(range(575, 580)), age_days=6.9)
    napi = FakeNumerAPI(_validation(range(575, 581)))

    assert not refresh_validation("5.3", napi=napi, now=NOW)

    assert napi.downloads == []
    assert _eras_on_disk(path) == 5


def test_a_copy_older_than_a_week_is_replaced(datasets_dir):
    path = _cache_validation(datasets_dir, _validation(range(575, 580)), age_days=7.1)
    napi = FakeNumerAPI(_validation(range(575, 581)))

    assert refresh_validation("5.3", napi=napi, now=NOW)

    assert napi.downloads == ["v5.3/validation.parquet"]
    assert _eras_on_disk(path) == 6
    assert _leftovers(datasets_dir) == []


def test_a_failed_download_keeps_the_old_copy_and_only_warns(datasets_dir):
    path = _cache_validation(datasets_dir, _validation(range(575, 580)), age_days=8)

    with pytest.warns(RuntimeWarning, match="validation.parquet refresh failed"):
        assert not refresh_validation("5.3", napi=FakeNumerAPI(None), now=NOW)

    assert _eras_on_disk(path) == 5


def test_a_download_that_is_not_a_readable_parquet_never_replaces_the_old_copy(datasets_dir):
    """The next run reads this file *before* it uploads, so a bad one would cost a submission."""
    path = _cache_validation(datasets_dir, _validation(range(575, 580)), age_days=8)

    with pytest.warns(RuntimeWarning, match="validation.parquet refresh failed"):
        assert not refresh_validation("5.3", napi=GarbageNumerAPI(None), now=NOW)

    assert _eras_on_disk(path) == 5
    assert _leftovers(datasets_dir) == []


def test_a_download_missing_the_target_never_replaces_the_old_copy(datasets_dir):
    path = _cache_validation(datasets_dir, _validation(range(575, 580)), age_days=8)
    napi = FakeNumerAPI(_validation(range(575, 581)).drop(columns="target"))

    with pytest.warns(RuntimeWarning, match="target"):
        assert not refresh_validation("5.3", napi=napi, now=NOW)

    assert _eras_on_disk(path) == 5


def test_an_interrupted_earlier_download_is_never_resumed(datasets_dir):
    """numerapi resumes any `<dest>.temp` without asking whether the server's file
    changed since, which would splice two versions into one file. Staging
    leftovers are deleted before every refresh."""
    _cache_validation(datasets_dir, _validation(range(575, 580)), age_days=8)
    staging = datasets_dir / "5.3" / "validation.parquet.new"
    Path(f"{staging}.temp").write_bytes(b"half of last week's file")
    staging.write_bytes(b"a staged file that never got swapped in")
    seen: list[bool] = []

    class Recording(FakeNumerAPI):
        def download_dataset(self, filename: str, dest_path: str) -> None:
            seen.append(Path(f"{dest_path}.temp").exists() or Path(dest_path).exists())
            super().download_dataset(filename, dest_path)

    assert refresh_validation("5.3", napi=Recording(_validation(range(575, 581))), now=NOW)

    assert seen == [False]
    assert _leftovers(datasets_dir) == []
