# Context

## Glossary

**Stage** — one step of the `zemir_0.1` pipeline (download, train, neutralize, submit). Stages run in a fixed order, driven by one entrypoint, exchanging data as in-memory `pandas` DataFrames within a single process.

**Model** — an object implementing the `Model` protocol (`zemir/models/base.py`): `fit(X, y, era, *, checkpoint_path=None) -> FitResult` and `predict(X) -> np.ndarray`. A structural `typing.Protocol`, not an ABC — linear regression and era-boosted XGBoost share no base-class behavior, only this shape. `predict()` is pure and stateless: raw, unneutralized, uncombined predictions, so multiple models' outputs can be combined for ensembling later without touching either implementation.

**FitResult** — the return value of `Model.fit()`: `{model, history}`, where `history` is `None` for models with no per-iteration record (linear regression) or a DataFrame of per-era scores over training iterations (era-boosting).

**RunConfig** — the single Python dataclass, composed of `FeatureConfig`, `EraBoostConfig | None`, `NeutralizationConfig`, plus `data_version` and `run_id`, that fully parameterizes one pipeline invocation. Built once per run and threaded down into every stage — not owned by a `Model` instance, since the same model class can run under different configs.

**Run** — one execution of the pipeline entrypoint, identified by `run_id` (an explicit name from `RunConfig`, or an auto-generated timestamp). Its outputs land under `prod/zemir_0.1/runs/<run_id>/`.

## Decision records

- [0001: zemir_0.1 pipeline architecture](docs/adr/0001-zemir-pipeline-architecture.md)
