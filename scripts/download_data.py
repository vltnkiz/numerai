"""Download Numerai datasets into datasets/{version}/ via NumerAPI.

Interactive usage (pick version, then files, from an in-terminal menu):
    python scripts/download_data.py

Scripted usage (skips the menus):
    python scripts/download_data.py --datasets features.json validation.parquet
    python scripts/download_data.py --version 5.1 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from numerapi import NumerAPI
from prompt_toolkit.shortcuts import button_dialog, checkboxlist_dialog, radiolist_dialog

REPO_ROOT = Path(__file__).resolve().parent.parent

AVAILABLE_DATASETS = [
    "features.json",
    "train.parquet",
    "train_benchmark_models.parquet",
    "validation.parquet",
    "validation_benchmark_models.parquet",
    "validation_example_preds.parquet",
    "validation_example_preds.csv",
    "live.parquet",
    "live_benchmark_models.parquet",
    "live_example_preds.parquet",
    "live_example_preds.csv",
    "meta_model.parquet",
]

DEFAULT_DATASETS = ["features.json", "validation.parquet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Numerai datasets into datasets/{version}/. "
        "Run with no arguments for an interactive menu."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Numerai data version, without the 'v' prefix (e.g. 5.0).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=AVAILABLE_DATASETS,
        default=None,
        help=f"Files to download (default: {DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


def list_versions(napi: NumerAPI) -> list[str]:
    files = napi.list_datasets()
    versions = {f.split("/", 1)[0][1:] for f in files if f.startswith("v")}
    return sorted(versions, key=lambda v: [int(p) for p in v.split(".")])


def list_files(napi: NumerAPI, version: str) -> list[str]:
    files = napi.list_datasets()
    prefix = f"v{version}/"
    return sorted(f[len(prefix) :] for f in files if f.startswith(prefix))


def prompt_for_selection(napi: NumerAPI) -> tuple[str, list[str]] | None:
    versions = list_versions(napi)
    version = radiolist_dialog(
        title="Numerai dataset download",
        text="Pick a data version:",
        values=[(v, v) for v in versions],
        default=versions[-1],
    ).run()
    if version is None:
        return None

    files = list_files(napi, version)
    default_files = [f for f in DEFAULT_DATASETS if f in files]
    selected = checkboxlist_dialog(
        title="Numerai dataset download",
        text=f"Pick files to download for v{version}:",
        values=[(f, f) for f in files],
        default_values=default_files,
    ).run()
    if not selected:
        return None

    return version, selected


def confirm_overwrite(existing: list[Path]) -> bool:
    names = "\n".join(f"  {p}" for p in existing)
    return button_dialog(
        title="Files already exist",
        text=f"These files already exist locally:\n{names}\n\nRe-download them?",
        buttons=[("Skip them", False), ("Re-download", True)],
    ).run()


def download_datasets(
    napi: NumerAPI, version: str, datasets: list[str], force: bool = False
) -> list[Path]:
    dest_dir = REPO_ROOT / "datasets" / version
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for filename in datasets:
        dest_path = dest_dir / filename
        if dest_path.exists() and not force:
            print(f"skip (exists): {dest_path}")
            downloaded.append(dest_path)
            continue
        remote_path = f"v{version}/{filename}"
        print(f"downloading {remote_path} -> {dest_path}")
        napi.download_dataset(remote_path, dest_path=str(dest_path))
        downloaded.append(dest_path)
    return downloaded


def main() -> None:
    args = parse_args()
    napi = NumerAPI()

    if args.version is None and args.datasets is None:
        selection = prompt_for_selection(napi)
        if selection is None:
            print("Cancelled.")
            sys.exit(0)
        version, datasets = selection

        dest_dir = REPO_ROOT / "datasets" / version
        existing = [dest_dir / f for f in datasets if (dest_dir / f).exists()]
        force = args.force
        if existing and not force:
            force = confirm_overwrite(existing)
    else:
        version = args.version or "5.0"
        datasets = args.datasets or DEFAULT_DATASETS
        force = args.force

    download_datasets(napi, version, datasets, force=force)


if __name__ == "__main__":
    main()
