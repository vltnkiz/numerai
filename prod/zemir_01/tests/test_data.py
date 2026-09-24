import pandas as pd
import pytest

from zemir import data
from zemir.data import load_meta_model


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
