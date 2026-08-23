"""Download stage: fetch train, validation, and current-round live data.

Ports scripts/download_data.py's NumerAPI usage. Caches train/validation/
features.json to the shared repo-root datasets/{version}/ directory (existing
behavior, reused as-is per docs/adr/0001-zemir-pipeline-architecture.md).

live.parquet is never served from cache: a new round's live features replace
the previous round's under the same filename, so a stale local copy would
silently feed the pipeline last round's data. See
docs/research/live-round-data-and-submission.md for the live-data shape
(era == "X", data_type == "live", target columns present but all-null) and
for why `id` arrives as the DataFrame index rather than a column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from numerapi import NumerAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"


@dataclass
class DownloadResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    live: pd.DataFrame
    feature_sets: dict[str, list[str]]


def _dataset_dir(version: str) -> Path:
    return DATASETS_DIR / version


def _remote_path(version: str, filename: str) -> str:
    return f"v{version}/{filename}"


def _download_file(
    napi: NumerAPI, version: str, filename: str, *, force: bool
) -> Path:
    dest_dir = _dataset_dir(version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists() and not force:
        return dest_path
    napi.download_dataset(_remote_path(version, filename), dest_path=str(dest_path))
    return dest_path


def _feature_columns(feature_sets: dict[str, list[str]], feature_set: str) -> list[str]:
    try:
        return feature_sets[feature_set]
    except KeyError:
        raise ValueError(
            f"unknown feature_set {feature_set!r}; available: {sorted(feature_sets)}"
        ) from None


def _read_parquet(path: Path, feature_sets: dict[str, list[str]], feature_set: str | None) -> pd.DataFrame:
    if feature_set is None:
        return pd.read_parquet(path)
    import pyarrow.parquet as pq

    all_columns = pq.read_schema(path).names
    non_feature = [c for c in all_columns if not c.startswith("feature_")]
    columns = non_feature + _feature_columns(feature_sets, feature_set)
    return pd.read_parquet(path, columns=columns)


def download(
    version: str,
    *,
    feature_set: str | None = None,
    force: bool = False,
    napi: NumerAPI | None = None,
) -> DownloadResult:
    """Fetch train, validation, and current-round live data for `version`.

    `feature_set`, if given, selects a key from features.json's feature_sets
    and restricts the loaded columns to it (plus era/data_type/target
    columns) — avoids materializing all ~2400+ feature columns in memory
    when a run only needs one feature set, per the memory concern flagged in
    ADR 0001's consequences.
    """
    napi = napi or NumerAPI()

    features_path = _download_file(napi, version, "features.json", force=force)
    feature_sets = json.loads(features_path.read_text())["feature_sets"]

    train_path = _download_file(napi, version, "train.parquet", force=force)
    validation_path = _download_file(napi, version, "validation.parquet", force=force)
    live_path = _download_file(napi, version, "live.parquet", force=True)

    train = _read_parquet(train_path, feature_sets, feature_set)
    validation = _read_parquet(validation_path, feature_sets, feature_set)
    live = _read_parquet(live_path, feature_sets, feature_set)

    return DownloadResult(
        train=train, validation=validation, live=live, feature_sets=feature_sets
    )
