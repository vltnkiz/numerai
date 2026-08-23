from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "datasets"


@dataclass
class Dataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    live: pd.DataFrame
    feature_columns: list[str]


def _download_file(napi: NumerAPI, version: str, filename: str, *, force: bool) -> Path:
    dest_dir = DATASETS_DIR / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists() and not force:
        return dest_path
    napi.download_dataset(f"v{version}/{filename}", dest_path=str(dest_path))
    return dest_path


def _read_parquet(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    all_columns = pq.read_schema(path).names
    non_feature = [c for c in all_columns if not c.startswith("feature_")]
    return pd.read_parquet(path, columns=non_feature + feature_columns)


def download(version: str, feature_set: str, *, napi: NumerAPI | None = None) -> Dataset:
    napi = napi or NumerAPI()

    features_path = _download_file(napi, version, "features.json", force=False)
    feature_columns = json.loads(features_path.read_text())["feature_sets"][feature_set]

    train_path = _download_file(napi, version, "train.parquet", force=False)
    validation_path = _download_file(napi, version, "validation.parquet", force=False)
    # live.parquet is never served from cache: a new round's features replace
    # the previous round's under the same filename.
    live_path = _download_file(napi, version, "live.parquet", force=True)

    return Dataset(
        train=_read_parquet(train_path, feature_columns),
        validation=_read_parquet(validation_path, feature_columns),
        live=_read_parquet(live_path, feature_columns),
        feature_columns=feature_columns,
    )
