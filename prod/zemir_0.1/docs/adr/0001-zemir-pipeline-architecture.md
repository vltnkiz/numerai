# 0001: zemir_0.1 pipeline architecture

## Status

Accepted

## Context

[Zemir Numerai Pipeline](https://github.com/vltnkiz/numerai/issues/1) needs a settled module shape before any stage (download, train, neutralize, submit) can be built, since the map requires a second model type (era-boosted XGBoost) to plug in later without touching the rest of the pipeline. The two reference implementations already in the repo have incompatible call shapes: linear regression is a plain `model.fit(X, y)` / `model.predict(X)`, while `era_boost_train_with_history(X, y, era_col, proportion, trees_per_step, num_iters, snapshot_every, checkpoint_path)` in `era-boosting/code/era_boosting.ipynb` takes era data and a checkpoint path, and returns a new `(model, era_scores, history_df)` tuple rather than mutating in place.

## Decision

- **`Model` is a structural `typing.Protocol`**, not an ABC: `fit(X, y, era, *, checkpoint_path=None) -> FitResult`, `predict(X) -> np.ndarray`. Every model type receives `era` and `checkpoint_path` even if it ignores them (linear regression does), so the pipeline can call every model type identically.
- **`fit` returns a `FitResult` (`{model, history}`)** rather than mutating `self`, matching era-boosting's actual return shape. `history` is `None` where there's nothing per-iteration to report.
- **`predict()` is pure and stateless** — raw `np.ndarray`, no neutralization or rank-normalization baked in — specifically so ensembling (deferred until a second model exists) only needs pipeline-level code to combine per-model prediction columns, not changes to either `Model` implementation.
- **Config is one `RunConfig` dataclass composed of nested sub-configs** (`FeatureConfig`, `EraBoostConfig | None`, `NeutralizationConfig`), built once per pipeline invocation and passed down into stages — not owned by the `Model` instance.
- **Stages exchange data as in-memory DataFrames within a single process**, not via intermediate files. `download.py` still caches to `datasets/{version}/*.parquet` on disk (existing behavior), and each stage persists its own output artifact to `runs/<run_id>/` as a side effect for resumability/inspection — but the pipeline itself doesn't round-trip through those files between stages.

## Consequences

- Every later ticket (dataset download, linear regression training, feature neutralization, submission, entrypoint wiring, second-model integration) builds directly against this shape; changing it after those land means touching all of them.
- Ensembling later is expected to need no `Model` protocol changes — only pipeline-level combination code — because `predict()` was constrained to stay pure from the start.
- Running everything in-memory in one process means the whole pipeline (a ~2.7M-row train set) must fit in the memory of whatever runs the weekly GitHub Action; this wasn't stress-tested against Actions runner limits during this decision and may need revisiting once ticket 10/11 (entrypoint wiring, scheduling) are worked.
