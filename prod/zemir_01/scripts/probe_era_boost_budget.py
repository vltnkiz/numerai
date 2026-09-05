#!/usr/bin/env python3
"""Checkpoint era_boost's own budget curve, and log worst-era churn (issue #36).

`fit_harness.py`/`score_harness.py` only ever compare *endpoints* — a full fit
at one `num_iters`. This instruments a single fit of the shipped config (or
`--max-eras`/`--num-iters`-overridden, for a fast correctness check) via
`train_xgboost`'s `on_iteration` hook: every iteration's worst-era set is
logged for free (already computed inside the loop), and every
`--checkpoint-every`'th iteration also gets scored against validation on
Numerai's paid metrics (`era_numerai_corr`, `era_mmc`) — the same functions
`zemir.harness` uses, so a checkpoint's numbers are directly comparable to any
harness run.

One training pass, not `num_iters / checkpoint_every` separate fits: the loop
is additive (`xgb_model=booster`), so a mid-loop checkpoint is exactly the
model a separate `--num-iters N` run would produce, without paying for N
redundant re-fits from scratch.

Usage:
  python scripts/probe_era_boost_budget.py                      # full scale, shipped config
  python scripts/probe_era_boost_budget.py --max-eras 120 --num-iters 6   # fast correctness check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from zemir.config import LIVE
from zemir.data import feature_columns as _feature_columns
from zemir.data import load_meta_model, load_split
from zemir.models import train_xgboost
from zemir.pipeline import RUNS_DIR, _last_eras
from zemir.scoring import era_mmc, era_numerai_corr

PROBE_DIR = RUNS_DIR / "probe_era_budget"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-eras", type=int, default=None)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    config = LIVE
    if args.max_eras is not None:
        config = replace(config, max_eras=args.max_eras)
    xgb_kwargs = dict(config.xgboost)
    if args.num_iters is not None:
        xgb_kwargs["num_iters"] = args.num_iters
    num_iters = xgb_kwargs["num_iters"]

    feature_columns = _feature_columns(config.data_version, config.feature_set)

    train_df = load_split(
        config.data_version,
        config.feature_set,
        "train",
        feature_names=feature_columns,
        target_column=config.target_column,
    ).dropna(subset=["target"])
    if config.max_eras is not None:
        train_df = _last_eras(train_df, config.max_eras)
    X, y, era = train_df[feature_columns], train_df["target"], train_df["era"]
    train_df = None

    validation_df = load_split(
        config.data_version,
        config.feature_set,
        "validation",
        feature_names=feature_columns,
        target_column=config.target_column,
    ).dropna(subset=["target"])
    if config.max_eras is not None:
        validation_df = _last_eras(validation_df, config.max_eras)

    meta_model = load_meta_model(config.data_version)

    checkpoints: list[dict] = []
    churn: list[dict] = []

    def on_iteration(iteration: int, predictor, worst_eras: list) -> None:
        churn.append({"iteration": iteration, "worst_eras": sorted(worst_eras, key=int)})
        is_last = iteration == num_iters
        if iteration != 0 and not is_last and iteration % args.checkpoint_every != 0:
            return

        preds = predictor.predict(validation_df[feature_columns])
        frame = pd.DataFrame({"prediction": preds}, index=validation_df.index)
        corr = era_numerai_corr(frame, validation_df["target"], validation_df["era"])
        mmc = era_mmc(frame, validation_df["target"], validation_df["era"], meta_model)
        row = {
            "iteration": iteration,
            "trees": iteration * xgb_kwargs["trees_per_step"],
            "mean_corr": float(corr["prediction"].mean()),
            "mean_mmc": float(mmc["prediction"].mean()) if len(mmc) else None,
        }
        checkpoints.append(row)
        mmc_str = f"{row['mean_mmc']:.6f}" if row["mean_mmc"] is not None else "n/a"
        print(
            f"[{iteration:>3}/{num_iters}] trees={row['trees']:>5}  "
            f"mean_corr={row['mean_corr']:.6f}  mean_mmc={mmc_str}"
        )

    train_xgboost(X, y, era, **xgb_kwargs, on_iteration=on_iteration)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PROBE_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2))
    (run_dir / "churn.json").write_text(json.dumps(churn, indent=2))
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_version": config.data_version,
                "feature_set": config.feature_set,
                "feature_count": len(feature_columns),
                "max_eras": config.max_eras,
                "xgboost": xgb_kwargs,
                "train_eras": len(era.unique()),
                "validation_eras": int(validation_df["era"].nunique()),
            },
            indent=2,
            sort_keys=True,
        )
    )

    non_zero_iterations = [c for c in churn if c["iteration"] > 0]
    if non_zero_iterations:
        all_sets = [set(c["worst_eras"]) for c in non_zero_iterations]
        locked = set.intersection(*all_sets)
        union = set.union(*all_sets)
        overlaps = [
            len(all_sets[i] & all_sets[i + 1]) / len(all_sets[i] | all_sets[i + 1])
            for i in range(len(all_sets) - 1)
        ]
        print(f"\nworst-era churn over {len(all_sets)} iterations:")
        print(f"  locked (in every iteration's worst set): {len(locked)} / {len(union)} ever-worst eras")
        print(
            f"  consecutive-iteration Jaccard overlap: "
            f"min={min(overlaps):.3f} mean={sum(overlaps) / len(overlaps):.3f} max={max(overlaps):.3f}"
        )

    print(f"\nrun_dir: {run_dir}")


if __name__ == "__main__":
    main()
