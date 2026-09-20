"""Score the pipeline **as actually submitted**, over validation eras.

`run_pipeline` scores raw per-model predictions — the artifact nobody uploads.
This harness scores the uploaded one: combine → neutralize → rank-normalize,
on Numerai's *paid* metrics (`numerai_corr` and MMC), so "is A better than B"
is answerable with a number.

Split in two stages, because fitting is the only expensive part:

1. `fit_validation_predictions` — trains each model on the train eras exactly as
   production does (same trainers, same frames, via `zemir.pipeline`), caches
   its raw validation predictions to parquet, and keeps each fitted model beside
   them (`load_models`) — the fit is the ~1h part, so anything that wants to ask
   the models a question (`explain_harness.py`) reads them rather than refitting.
2. `score_configs` — sweeps blends and neutralizations over those cached
   predictions, each row a `Strategy` scored through `zemir.blend`: the same
   transform the live run applies. Seconds, and repeatable without refitting.

Anything that changes a *model* (hyperparameters, feature set, training eras)
invalidates the cache and needs a new fit. Anything downstream of `.predict()`
is free.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from zemir.blend import blend_strategies, combine_predictions
from zemir.config import STRATEGIES, PipelineConfig
from zemir.data import (
    load_meta_model,
    load_validation_features,
    resolve_feature_sets,
    scoring_window,
)
from zemir.fitting import fit_strategy
from zemir.models import FittedModel
from zemir.pipeline import RUNS_DIR, predict_each
from zemir.scoring import (
    era_feature_corr,
    era_max_feature_corr,
    era_mmc,
    era_numerai_corr,
    summarize_era_scores,
)
from zemir.strategy import (
    BlendSpec,
    FeatureSets,
    ModelSpec,
    Strategy,
    neutralizer_columns,
    strategy_from_record,
    union_columns,
)

HARNESS_DIR = RUNS_DIR / "harness"
PREDICTIONS_FILENAME = "validation_predictions.parquet"
MODELS_DIRNAME = "models"  # <run_dir>/models/<ModelSpec.name>/, one FittedModel.save each


@dataclass
class FitResult:
    run_dir: Path
    predictions_path: Path
    predictions: pd.DataFrame


def fit_validation_predictions(
    config: PipelineConfig,
    strategy: Strategy,
    *,
    run_id: str,
    feature_sets: FeatureSets,
    load_train: Callable[[], pd.DataFrame],
    load_validation: Callable[[], pd.DataFrame],
    harness_dir: Path = HARNESS_DIR,
) -> FitResult:
    """Fit `strategy`'s models and cache their raw validation predictions.

    The expensive stage. Writes `validation_predictions.parquet` (index `id`;
    columns `era`, `target`, one per model) plus the `fit_config.json` that
    produced it — a scoring run stamps that config onto its results so a
    comparison is never read against the wrong fit — and, under `models/`, each
    fitted model itself. The models are written the moment the fit returns, before
    validation is loaded: they are tiny (a booster, or ~3,414 coefficients), and
    a failure past this point should not cost the fit. A run that dies before the
    parquet lands is still not a cache (`latest_cache` looks for the parquet).

    `strategy` names its own trainers via `ModelSpec.trainer`, resolved
    through `zemir.strategy.TRAINERS` — trying a regularized linear stage
    (issue #38) or a chunked one (issue #50) is a caller building its own
    `Strategy` with `linear`'s spec swapped for a different trainer name,
    rather than overriding a `build_trainers` callable.

    Takes `feature_sets` and a `load_train`/`load_validation` pair of
    zero-argument callables rather than a `Dataset` (contrast
    `zemir.pipeline.run_pipeline`, which does take one): the caller builds
    each loader around its own I/O (a real `load_split` call bound to
    `union_columns(strategy, feature_sets)` — already narrowed for issue #45's
    dropped columns before either loader is ever called — in production; a
    synthetic fixture in a test), but *when* each is called stays right here,
    one at a time — train first, fitted and freed inside `fit_strategy`
    (`zemir.fitting`, which owns that discipline), then validation — so a
    `Dataset` argument that forced both splits resident up front could never
    silently undo issue #47/#51's memory discipline.
    """
    run_dir = harness_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fit = fit_strategy(strategy, load_train, feature_sets=feature_sets, max_eras=config.max_eras)

    for name, model in fit.models.items():
        model.save(run_dir / MODELS_DIRNAME / name)

    validation_df = scoring_window(load_validation(), config.max_eras)

    predictions = predict_each(fit.models, validation_df)

    frame = validation_df[["era", "target"]].copy()
    for name, preds in predictions.items():
        frame[name] = preds.astype("float32")

    predictions_path = run_dir / PREDICTIONS_FILENAME
    frame.to_parquet(predictions_path)
    (run_dir / "fit_config.json").write_text(
        json.dumps(
            {
                "data_version": config.data_version,
                "feature_set": config.feature_set,
                "feature_count": len(union_columns(strategy, feature_sets)),
                "max_eras": config.max_eras,
                "strategy": asdict(strategy),
                "train_eras": fit.train_eras,
                "validation_eras": len(frame["era"].unique()),
                "validation_rows": len(frame),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return FitResult(run_dir=run_dir, predictions_path=predictions_path, predictions=frame)


class NoPersistedModels(Exception):
    """The cache was fitted before models were kept beside it — nothing to load, and no way to get them but a refit."""


def load_models(run_dir: Path) -> dict[str, FittedModel]:
    """Every model a cache was fitted with, in the strategy's declared order.

    Raises `NoPersistedModels` for a cache that predates persistence (it has no
    `models/` directory). Any other problem — a model the cache's own
    `fit_config.json` names but `models/` lacks, an interrupted or
    unreadable save — is an ordinary error naming the path, never a partial
    dict and never a refit: a cache that claims models and cannot produce them
    is broken, which is a different thing from an old one.
    """
    models_dir = run_dir / MODELS_DIRNAME
    if not models_dir.is_dir():
        raise NoPersistedModels(
            f"{run_dir} has no {MODELS_DIRNAME}/ directory: it was fitted before models "
            "were kept, so they would have to be refitted (scripts/fit_harness.py)"
        )
    fit = json.loads((run_dir / "fit_config.json").read_text())
    names = [spec["name"] for spec in fit["strategy"]["models"]]
    missing = [name for name in names if not (models_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"{models_dir} lacks the model(s) {missing} that {run_dir}'s fit names")
    return {name: FittedModel.load(models_dir / name) for name in names}


def explain_models(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Each cached model's `explain()`, ranked most-leaned-on first.

    The ranking is the only thing added to what `FittedModel.explain` returns,
    and it is a choice: a linear model by `|coef|` (the sign is the direction,
    not the size), a booster by `gain` — the mean loss reduction per split, and
    the one of its quantities that reproduces best across seeds (issue #83;
    `total_gain` reproduces worst, which is why `explain` leaves it out). A
    booster feature it never split on has no gain and sorts last. Ties keep
    column order, so the same cache always prints the same table.
    """
    ranked = {}
    for name, model in load_models(run_dir).items():
        explanation = model.explain()
        if "gain" in explanation:
            ranked[name] = explanation.sort_values("gain", ascending=False, kind="stable", na_position="last")
        else:
            ranked[name] = explanation.sort_values("coef", key=abs, ascending=False, kind="stable")
    return ranked


@dataclass
class ScoreResult:
    run_dir: Path
    summary: pd.DataFrame
    era_corr: pd.DataFrame
    era_mmc: pd.DataFrame
    era_max_feature_corr: pd.DataFrame


@dataclass(frozen=True)
class FitRecord:
    """What a cache was fitted with, as far as its own `fit_config.json` can say.

    `verified` is whether that is a *record* or a reconstruction. A cache fitted
    since #74 stores the whole `Strategy`, and `check_scoreable` can then hold a
    swept strategy's fit fields (`features`, `trainer`, `params`) to it — also
    when the record predates `Neutralization`, since its fit fields are intact.
    The oldest caches store only the model names, so `strategy` is rebuilt from
    `zemir.config.STRATEGIES` by name and nothing about how those models were
    fitted can be checked — the only claim is that the names are in the cache.
    """

    strategy: Strategy
    feature_set: str
    verified: bool


class UnscoreableStrategy(ValueError):
    """A strategy this cache cannot be rescored under, without a refit."""


def _recorded_strategy(fit: Mapping[str, object]) -> Strategy | None:
    if "strategy" not in fit:
        return None
    try:
        return strategy_from_record(fit["strategy"])
    except KeyError:
        # Recorded before `Neutralization` existed, so its blend and per-model
        # neutralizers are in the old vocabulary. Those are not fit fields, which
        # are all a scorer holds a swept strategy to, and they are still there.
        return Strategy(
            models=tuple(
                ModelSpec(
                    name=m["name"],
                    features=m["features"],
                    trainer=m["trainer"],
                    params=dict(m["params"]),
                )
                for m in fit["strategy"]["models"]
            ),
            blend=BlendSpec(),
        )


def load_fit_record(run_dir: Path) -> FitRecord:
    """The `Strategy` a cache was fitted with, and whether that is on record or reconstructed."""
    fit = json.loads((run_dir / "fit_config.json").read_text())
    recorded = _recorded_strategy(fit)
    if recorded is not None:
        return FitRecord(recorded, fit["feature_set"], verified=True)

    known: dict[str, ModelSpec] = {}
    for strategy in STRATEGIES.values():
        for spec in strategy.models:
            known.setdefault(spec.name, spec)
    unknown = [name for name in fit["models"] if name not in known]
    if unknown:
        raise KeyError(
            f"{run_dir} predates recorded strategies and names model(s) {unknown} that no "
            f"strategy in zemir.config.STRATEGIES defines (have: {sorted(known)})"
        )
    # In `STRATEGIES`' order, not the cache's (which lists them sorted): a sweep names its
    # rows from the model order, and `linear+era_boost_p0.5` is what every table has called it.
    reconstructed = Strategy(
        models=tuple(spec for name, spec in known.items() if name in fit["models"]),
        blend=BlendSpec(),
    )
    return FitRecord(reconstructed, fit["feature_set"], verified=False)


def _fit_fields(spec: ModelSpec) -> dict[str, object]:
    # Through JSON so a tuple and the list it round-trips to compare equal.
    return {
        "features": spec.features,
        "trainer": spec.trainer,
        "params": json.loads(json.dumps(dict(spec.params), sort_keys=True)),
    }


def check_scoreable(strategy: Strategy, record: FitRecord) -> None:
    """Raise `UnscoreableStrategy` unless `strategy` can be scored against this cache.

    Every model it names must have been fitted and — for a cache that recorded
    its strategy — fitted the way the strategy says: same `features`, `trainer`
    and `params`. Weights and neutralization are not checked, because they are
    exactly what a rescore is for. Scoring a strategy whose `params` differ from
    the fit would report the cached model's numbers under the wrong description.
    """
    fitted = {spec.name: spec for spec in record.strategy.models}
    missing = [spec.name for spec in strategy.models if spec.name not in fitted]
    if missing:
        raise UnscoreableStrategy(
            f"model(s) {missing} were not fitted in this cache (it holds {sorted(fitted)})"
        )
    if not record.verified:
        return
    differences = []
    for spec in strategy.models:
        swept, cached = _fit_fields(spec), _fit_fields(fitted[spec.name])
        differences += [
            f"{spec.name}.{field}: {swept[field]!r} here, {cached[field]!r} in the cache"
            for field in swept
            if swept[field] != cached[field]
        ]
    if differences:
        raise UnscoreableStrategy(
            "fitted differently from what this cache holds, which needs a refit: "
            + "; ".join(differences)
        )


def unscoreable_reason(strategy: Strategy, record: FitRecord) -> str | None:
    """Why `strategy` cannot be scored against this cache, or `None` if it can."""
    try:
        check_scoreable(strategy, record)
    except UnscoreableStrategy as exc:
        return str(exc)
    return None


def _load_cache_and_features(
    run_dir: Path, strategies: Mapping[str, Strategy] | None = None
) -> tuple[pd.DataFrame, dict, pd.DataFrame, list[str], FeatureSets]:
    """The cache, its own fit config, validation features, the scoring universe, and feature sets.

    The features are the cache's `feature_set` (the universe exposure is
    measured against, whatever a strategy neutralized on) plus every column any
    of `strategies` neutralizes against. Shared by `score_configs` and
    `rank_feature_exposure` so both read the dataset the cache was actually fit
    on, never a guessed one.
    """
    strategies = strategies or {}
    cache = pd.read_parquet(run_dir / PREDICTIONS_FILENAME)
    fit = json.loads((run_dir / "fit_config.json").read_text())
    eras = sorted(cache["era"].unique(), key=int)
    names = [fit["feature_set"], *(n for s in strategies.values() for n in s.feature_set_names)]
    feature_sets = resolve_feature_sets(fit["data_version"], names)
    scoring_universe = list(feature_sets[fit["feature_set"]])
    wanted = dict.fromkeys(
        [
            *scoring_universe,
            *(
                c
                for s in strategies.values()
                for n in s.neutralizations
                for c in neutralizer_columns(n, feature_sets)
            ),
        ]
    )
    features = load_validation_features(fit["data_version"], list(wanted), eras=eras).loc[
        cache.index
    ]
    return cache, fit, features, scoring_universe, feature_sets


def rank_feature_exposure(
    run_dir: Path,
    *,
    models: tuple[str, ...] = ("linear", "era_boost"),
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Mean absolute exposure of every feature to one blend, across every validation era.

    Answers "exposure to what" for issue #35: ranks the pre-neutralization
    blend — the artifact neutralization actually corrects — so a deliberate
    neutralizer subset can target the features it is measurably most exposed
    to, rather than a guessed or merely-published group. Descending order:
    `.index[:k]` is the top-k most-exposed feature set.
    """
    cache, _, features, scoring_universe, _ = _load_cache_and_features(run_dir)
    blend = combine_predictions({name: cache[name] for name in models}, cache["era"], weights)
    exposure = era_feature_corr(blend, features[scoring_universe], cache["era"])
    return exposure.mean().sort_values(ascending=False)


def score_configs(
    strategies: Mapping[str, Strategy],
    *,
    run_dir: Path,
) -> ScoreResult:
    """Score every strategy against one cached fit, on Numerai's paid metrics.

    The cheap stage: hands the cache's raw per-model predictions to
    `zemir.blend.blend_strategies` — the very transform the live run applies to
    the single live era — and scores the result. `rank_normalize` is
    deliberately *not* applied: it is a strictly monotone transform, and CORR,
    MMC and the exposure measure here all rank internally, so it is provably
    free (issue #26).

    Each row is a `Strategy`, and must be one this cache can be scored under
    (`check_scoreable`): the models are the cache's, so only the blend and the
    neutralizations may differ from what was fitted. Reads which dataset produced
    the cache from its own `fit_config.json`, so a sweep can never be scored
    against the wrong features or the wrong meta model.

    The `max_feature_corr` diagnostic always measures exposure to the cache's
    whole `feature_set` regardless of what a strategy neutralized against — it
    is the total risk being reported on, not a number a subset gets to grade
    itself on.
    """
    record = load_fit_record(run_dir)
    for label, strategy in strategies.items():
        try:
            check_scoreable(strategy, record)
        except UnscoreableStrategy as exc:
            raise UnscoreableStrategy(f"{label}: {exc}") from None

    cache, fit, features, scoring_universe, feature_sets = _load_cache_and_features(
        run_dir, strategies
    )
    meta_model = load_meta_model(fit["data_version"])

    predictions = blend_strategies(strategies, cache, features, cache["era"], feature_sets).astype(
        "float32"
    )

    corr_by_era = era_numerai_corr(predictions, cache["target"], cache["era"])
    mmc_by_era = era_mmc(predictions, cache["target"], cache["era"], meta_model)
    exposure_by_era = era_max_feature_corr(predictions, features[scoring_universe], cache["era"])
    summary = summarize_era_scores(corr_by_era, mmc_by_era, exposure_by_era)

    summary.to_csv(run_dir / "scores.csv")
    corr_by_era.to_csv(run_dir / "era_corr.csv")
    mmc_by_era.to_csv(run_dir / "era_mmc.csv")
    exposure_by_era.to_csv(run_dir / "era_max_feature_corr.csv")
    (run_dir / "scoring_strategies.json").write_text(
        json.dumps(
            {
                "fit_fields": (
                    "verified"
                    if record.verified
                    else "unverified: this cache predates recorded strategies"
                ),
                "strategies": {label: asdict(strategy) for label, strategy in strategies.items()},
            },
            indent=2,
        )
    )
    return ScoreResult(
        run_dir=run_dir,
        summary=summary,
        era_corr=corr_by_era,
        era_mmc=mmc_by_era,
        era_max_feature_corr=exposure_by_era,
    )
