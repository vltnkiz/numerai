from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
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
    """Every non-feature column plus the requested features, and nothing else.

    Releases pyarrow's pool before returning: `pd.read_parquet` decodes
    through an intermediate Arrow table that pandas copies out of, and the
    now-unreferenced table's arena isn't returned to the OS on its own — the
    same pattern `load_validation_features`/`fit_models`/
    `fit_validation_predictions` already work around, measured elsewhere at
    `all` width as a ~2x RSS-vs-logical-size gap (issue #52). This is the one
    caller of `pd.read_parquet` in this module that lacked the fix (issue
    #54 found it while sizing a `max_eras`-truncated proxy fit at `all`
    width: `load_split` reads a split's *full* row count before any era
    truncation happens downstream, so this retained arena is paid in full
    regardless of `max_eras`, and was quietly already part of the untruncated
    full-scale peak).
    """
    all_columns = pq.read_schema(path).names
    non_feature = [c for c in all_columns if not c.startswith("feature_")]
    frame = pd.read_parquet(path, columns=non_feature + feature_columns)
    pa.default_memory_pool().release_unused()
    return frame


def feature_columns(
    version: str, feature_set: str, *, napi: NumerAPI | None = None
) -> list[str]:
    napi = napi or NumerAPI()
    features_path = _download_file(napi, version, "features.json", force=False)
    return json.loads(features_path.read_text())["feature_sets"][feature_set]


def load_split(
    version: str,
    feature_set: str,
    split: str,
    *,
    feature_names: list[str] | None = None,
    napi: NumerAPI | None = None,
) -> pd.DataFrame:
    """One dataset split ("train"/"validation"/"live"), and nothing else.

    `download` reads all three splits before returning, so any caller of it
    holds train+validation+live simultaneously even if it only ever touches
    one at a time — issue #47 found that eager triple-load, not the fit
    itself, is what pushes `all` width's ~3,414-column design matrix past the
    ~52GB WSL memory cap. The harness never touches `live`, and only needs
    train and validation one after the other, so it loads (and frees) them
    one split at a time via this instead.

    `feature_names` lets a caller pass an already-narrowed column list (e.g.
    issue #45's dead-column drop) straight through to the parquet read,
    rather than reading every column and dropping some afterward — dropping
    after the fact means briefly holding both the wide and narrow copies.
    """
    napi = napi or NumerAPI()
    names = feature_names if feature_names is not None else feature_columns(
        version, feature_set, napi=napi
    )
    path = _download_file(napi, version, f"{split}.parquet", force=(split == "live"))
    return _read_parquet(path, names)


def load_validation_features(
    version: str,
    feature_set: str,
    *,
    eras: list[str] | None = None,
    napi: NumerAPI | None = None,
) -> pd.DataFrame:
    """`era` plus the feature columns, and nothing else.

    Scoring needs validation features to neutralize against, but must not pay to
    load train and live the way `download` does — those are the fit's business,
    and the fit is already over by the time anything is scored. `eras` restricts
    the read at the parquet level, so a smoke-scale cache costs smoke-scale memory.

    Releases pyarrow's pool before returning: at `all` width this read measured
    29.67GiB resident against a 13.81GiB logical frame (issue #52) — the same
    mimalloc-retained-arena pattern `zemir.harness`/`zemir.models` already work
    around elsewhere, here because `pd.read_parquet` decodes through an
    intermediate Arrow table that pandas copies out of, and the now-unreferenced
    table's arena isn't returned to the OS on its own. This is what let
    `score_configs`'s subsequent `pd.concat` (building the model/blend
    projections) breach the machine's ~50GiB cap in under a minute, before any
    of the actual per-era neutralization math had run.
    """
    napi = napi or NumerAPI()
    columns = feature_columns(version, feature_set, napi=napi)
    path = _download_file(napi, version, "validation.parquet", force=False)
    filters = [("era", "in", list(eras))] if eras is not None else None
    frame = pd.read_parquet(path, columns=["era"] + columns, filters=filters)
    pa.default_memory_pool().release_unused()
    return frame


def load_meta_model(version: str, *, napi: NumerAPI | None = None) -> pd.Series:
    """Numerai's stake-weighted crowd prediction — the reference MMC is measured against.

    Covers a *window* of validation eras, not all of them (96 eras in v5.0), so
    anything scored against it is restricted to that window.
    """
    napi = napi or NumerAPI()
    path = _download_file(napi, version, "meta_model.parquet", force=False)
    frame = pd.read_parquet(path, columns=["numerai_meta_model"])
    return frame["numerai_meta_model"].dropna()


def download(version: str, feature_set: str, *, napi: NumerAPI | None = None) -> Dataset:
    napi = napi or NumerAPI()

    feature_names = feature_columns(version, feature_set, napi=napi)

    train_path = _download_file(napi, version, "train.parquet", force=False)
    validation_path = _download_file(napi, version, "validation.parquet", force=False)
    # live.parquet is never served from cache: a new round's features replace
    # the previous round's under the same filename.
    live_path = _download_file(napi, version, "live.parquet", force=True)

    return Dataset(
        train=_read_parquet(train_path, feature_names),
        validation=_read_parquet(validation_path, feature_names),
        live=_read_parquet(live_path, feature_names),
        feature_columns=feature_names,
    )
