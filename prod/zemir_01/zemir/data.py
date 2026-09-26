from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numerapi import NumerAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "datasets"
# Numerai resolves about one validation era a week, so a daily 5.6 GB pull
# would mostly fetch identical bytes.
VALIDATION_MAX_AGE = timedelta(days=7)


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


def _read_parquet(
    path: Path, feature_columns: list[str], *, target_column: str = "target"
) -> pd.DataFrame:
    """Every non-feature column plus the requested features, and nothing else.

    Releases pyarrow's pool before returning: `pd.read_parquet` decodes
    through an intermediate Arrow table that pandas copies out of, and the
    now-unreferenced table's arena isn't returned to the OS on its own — the
    same pattern `load_validation_features`/`fit_strategy`/
    `fit_validation_predictions` already work around, measured elsewhere at
    `all` width as a ~2x RSS-vs-logical-size gap (issue #52). This is the one
    caller of `pd.read_parquet` in this module that lacked the fix (issue
    #54 found it while sizing a `max_eras`-truncated proxy fit at `all`
    width: `load_split` reads a split's *full* row count before any era
    truncation happens downstream, so this retained arena is paid in full
    regardless of `max_eras`, and was quietly already part of the untruncated
    full-scale peak).

    `target_column` overwrites the dataset's generic `target` alias with a
    named target column (e.g. `target_ender_20`) — every non-feature column is
    already read above, so the named column is already present in `frame` and
    this is a same-frame reassignment, not another read. The alias is *not*
    guaranteed to track whichever target Numerai currently pays on (issue
    #64: v5.3's `target` aliases `target_ender_60`, not the `target_ender_20`
    Numerai has scored payouts against since 2026-01-01) — see CONTEXT.md.
    """
    all_columns = pq.read_schema(path).names
    non_feature = [c for c in all_columns if not c.startswith("feature_")]
    frame = pd.read_parquet(path, columns=non_feature + feature_columns)
    if target_column != "target" and "target" in frame.columns:
        frame["target"] = frame[target_column]
    pa.default_memory_pool().release_unused()
    return frame


def feature_columns(
    version: str, feature_set: str, *, napi: NumerAPI | None = None
) -> list[str]:
    napi = napi or NumerAPI()
    features_path = _download_file(napi, version, "features.json", force=False)
    return json.loads(features_path.read_text())["feature_sets"][feature_set]


def resolve_feature_sets(
    version: str,
    names: Iterable[str],
    *,
    drop: frozenset[str] = frozenset(),
    napi: NumerAPI | None = None,
) -> dict[str, list[str]]:
    """Each named feature set's columns, in `features.json`'s own order.

    The one place a `ModelSpec.features` name becomes columns. `drop` removes
    columns from *every* set (issue #45's dead-column drop), so a caller
    narrowing `all` width does it once here rather than per model.
    """
    return {
        name: [c for c in feature_columns(version, name, napi=napi) if c not in drop]
        for name in dict.fromkeys(names)
    }


def last_eras(df: pd.DataFrame, n: int) -> pd.DataFrame:
    eras = sorted(df["era"].unique(), key=int)[-n:]
    return df[df["era"].isin(eras)]


def scoring_window(df: pd.DataFrame, max_eras: int | None) -> pd.DataFrame:
    """The target-bearing rows of `df`, restricted to its last `max_eras` eras.

    What every fit and every validation score is computed over — one
    definition, so a train frame and a validation frame can never be
    windowed differently.
    """
    df = df.dropna(subset=["target"])
    return df if max_eras is None else last_eras(df, max_eras)


def load_split(
    version: str,
    feature_set: str,
    split: str,
    *,
    feature_names: list[str] | None = None,
    target_column: str = "target",
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

    `target_column` is passed straight through to `_read_parquet` (issue #64).
    """
    napi = napi or NumerAPI()
    names = feature_names if feature_names is not None else feature_columns(
        version, feature_set, napi=napi
    )
    path = _download_file(napi, version, f"{split}.parquet", force=(split == "live"))
    return _read_parquet(path, names, target_column=target_column)


def load_validation_features(
    version: str,
    columns: Sequence[str],
    *,
    eras: list[str] | None = None,
    napi: NumerAPI | None = None,
) -> pd.DataFrame:
    """`era` plus the requested feature columns, and nothing else.

    Takes columns, not a feature-set name, for the same reason `download` does:
    what scoring needs is the union of the run's scoring universe and every
    column a strategy's neutralizations name, which no single set describes.

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
    path = _download_file(napi, version, "validation.parquet", force=False)
    filters = [("era", "in", list(eras))] if eras is not None else None
    frame = pd.read_parquet(path, columns=["era", *columns], filters=filters)
    pa.default_memory_pool().release_unused()
    return frame


def load_meta_model(
    version: str, *, napi: NumerAPI | None = None, refresh: bool = False
) -> pd.Series:
    """Numerai's stake-weighted crowd prediction — the reference MMC is measured against.

    Covers a *window* of validation eras, not all of them (96 eras in v5.0), so
    anything scored against it is restricted to that window. Numerai appends an
    era a week and never revises one, so a stale copy is a shorter window, never
    wrong numbers.

    `refresh` re-downloads the shared on-disk copy. Only the live run sets it,
    after submitting; the harness reads whatever is on disk, so the daily run is
    what keeps both paths on the same window (issue #101). A failed refresh falls
    back to the cached copy with a warning rather than losing MMC for the run;
    numerapi writes to a `.temp` file and swaps it in, so a failure mid-download
    leaves that copy intact.
    """
    napi = napi or NumerAPI()
    try:
        path = _download_file(napi, version, "meta_model.parquet", force=refresh)
    except Exception as exc:
        path = DATASETS_DIR / version / "meta_model.parquet"
        if not path.exists():
            raise
        warnings.warn(
            f"meta_model.parquet refresh failed ({exc!r}); scoring against the stale cached copy",
            RuntimeWarning,
            stacklevel=2,
        )
    frame = pd.read_parquet(path, columns=["numerai_meta_model"])
    return frame["numerai_meta_model"].dropna()


def refresh_validation(
    version: str,
    *,
    napi: NumerAPI | None = None,
    max_age: timedelta = VALIDATION_MAX_AGE,
    now: datetime | None = None,
) -> bool:
    """Re-download validation.parquet when the on-disk copy is older than `max_age`.

    Numerai keeps resolving validation eras (targets arrive ~4 weeks after an
    era), so a frozen copy stops the gate's frame and the meta-model window at
    whatever it held the day it was fetched. Only the live run calls this, as
    its very last step, after the upload and the score log: the next run's
    gate reads the new copy, and every later harness fit does too. An old
    harness cache is unaffected — it carries its own eras and targets and
    reads features by id.

    Downloads to a staging file and swaps it in only once its footer reads
    and carries `era` and `target`, because the next run reads this file
    *before* it uploads: a broken copy there would cost a submission. Staging
    leftovers are deleted first — numerapi resumes any `<dest>.temp` without
    checking that the server's file is still the same one, which would splice
    two versions together. Any failure warns and keeps the old copy; returns
    whether the copy was replaced.
    """
    path = DATASETS_DIR / version / "validation.parquet"
    now = now or datetime.now(timezone.utc)
    if path.exists():
        age = now - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if age < max_age:
            return False
    staging = path.with_name(path.name + ".new")
    temp = staging.with_name(staging.name + ".temp")
    try:
        staging.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        (napi or NumerAPI()).download_dataset(
            f"v{version}/validation.parquet", dest_path=str(staging)
        )
        missing = {"era", "target"} - set(pq.read_schema(staging).names)
        if missing:
            raise ValueError(f"downloaded copy lacks {sorted(missing)}")
        staging.replace(path)
    except Exception as exc:
        warnings.warn(
            f"validation.parquet refresh failed ({exc!r}); keeping the existing copy",
            RuntimeWarning,
            stacklevel=2,
        )
        for leftover in (staging, temp):
            leftover.unlink(missing_ok=True)
        return False
    return True


def download(
    version: str,
    feature_names: list[str],
    *,
    target_column: str = "target",
    napi: NumerAPI | None = None,
) -> Dataset:
    """All three splits at `feature_names` width.

    Takes columns, not a feature-set name: a run's width is the union of what
    its models train on (plus every column a neutralization names), which no
    single named set describes — see `zemir.strategy.required_columns`.
    """
    napi = napi or NumerAPI()

    train_path = _download_file(napi, version, "train.parquet", force=False)
    validation_path = _download_file(napi, version, "validation.parquet", force=False)
    # live.parquet is never served from cache: a new round's features replace
    # the previous round's under the same filename.
    live_path = _download_file(napi, version, "live.parquet", force=True)

    return Dataset(
        train=_read_parquet(train_path, feature_names, target_column=target_column),
        validation=_read_parquet(
            validation_path, feature_names, target_column=target_column
        ),
        live=_read_parquet(live_path, feature_names, target_column=target_column),
        feature_columns=feature_names,
    )
