"""From per-model predictions to the blended, neutralized prediction — once, for both callers.

`run_pipeline` (the live run) and `zemir.harness.score_configs` (the sweep)
both turn "each model's raw predictions" into "what gets scored or submitted".
They used to do it twice, in different shapes: the live run neutralized the
blend only, the harness neutralized either the blend or every model uniformly,
and neither could neutralize one model differently from another. This module is
the one place that does it, driven by a `Strategy` (issue #88):

    per-model neutralization  ->  combine (rank-blend)  ->  blend neutralization

It stops there. `rank_normalize` is submission formatting, not part of the
transform: the live run applies it to what this returns, and the harness does
not, because CORR, MMC and the exposure measure all rank internally (issue #26).

The interface takes a *mapping* of strategies, not one, and that is the point:
a sweep scores dozens of rows against one cache, and the expensive part — a
least-squares solve per era — depends only on which column is being projected
and onto which features, never on the proportion or the blend weights. Solving
once per (column, feature set) and reusing it across every row that shares
them is the difference between minutes and an hour (issue #35). The live run is
the one-strategy case of the same call.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from zemir.strategy import FeatureSets, Neutralization, Strategy, neutralizer_columns


def combine_predictions(
    predictions: dict[str, pd.Series],
    era: pd.Series,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Weighted average of each model's predictions, ranked per era first.

    `weights` need not sum to 1 — it is normalized internally — and `None`
    means equal weight, so unweighted callers see identical behavior to before.

    A lone model is returned untouched rather than ranked. That is not a
    shortcut: whatever happens next — neutralization above all — sees the raw
    prediction, and ranking first would change the result. Keeping the rule here
    means the harness and the pipeline cannot disagree about what "combined" means.
    """
    if len(predictions) == 1:
        return next(iter(predictions.values())).rename("prediction")
    ranked = {name: preds.groupby(era).rank(pct=True) for name, preds in predictions.items()}
    if weights is None:
        combined = sum(ranked.values()) / len(ranked)
    else:
        combined = sum(ranked[name] * weights[name] for name in ranked) / sum(
            weights[name] for name in ranked
        )
    return combined.rename("prediction")


def project(scores: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """The linear component of `scores` explained by `exposures` (plus an intercept).

    Solved with `lstsq` rather than by forming `pinv(exposures)`: both take the
    minimum-norm least-squares solution and agree to ~1e-15, but lstsq never
    materializes the pseudo-inverse and is roughly twice as fast — which matters
    because a sweep solves this once per era.
    """
    exposures = np.hstack((exposures, np.ones((len(exposures), 1))))
    return exposures @ np.linalg.lstsq(exposures, scores, rcond=None)[0]


def era_feature_projection(
    df: pd.DataFrame,
    columns: list[str],
    neutralizers: list[str],
    *,
    era_col: str = "era",
) -> pd.DataFrame:
    """Per-era linear component of every column in `columns`, explained by `neutralizers`.

    Neutralizing at proportion `p` is `scores - p * projection` — affine in `p`,
    and the projection does not depend on `p` at all. So a sweep over proportions
    and blends costs one least-squares solve per era rather than one per
    configuration, which is the difference between minutes and an hour.

    Grouped per era rather than via `groupby().apply()`: pandas collapses a
    single-group `apply()` result into a DataFrame instead of a Series, and live
    data is always a single era.
    """
    parts = [
        pd.DataFrame(
            project(era_df[columns].to_numpy(dtype=float), era_df[neutralizers].values),
            index=era_df.index,
            columns=columns,
        )
        for _, era_df in df.groupby(era_col, group_keys=False)
    ]
    return pd.concat(parts).loc[df.index]


# A neutralization with its feature set resolved to columns: (proportion, columns).
_Resolved = tuple[float, tuple[str, ...]]


def _resolve(neutralization: Neutralization | None, feature_sets: FeatureSets) -> _Resolved | None:
    if neutralization is None:
        return None
    return neutralization.proportion, neutralizer_columns(neutralization, feature_sets)


def _projections(
    source: pd.DataFrame,
    groups: dict[tuple[str, ...], set[str]],
    features: pd.DataFrame,
    era: pd.Series,
) -> dict[tuple[str, tuple[str, ...]], pd.Series]:
    """Each source column's per-era projection onto each feature set that asks for it.

    Grouped by feature set, not computed one column at a time: every column
    projected onto the same features goes through one `era_feature_projection`
    call, so the per-era lstsq factorization is solved once and reused across
    columns — computed per column instead, this was ~3x slower on the harness's
    15-row sweep (issue #35). Only the columns a group names are concatenated,
    never the whole feature frame, since at `all` width that frame is the memory
    peak (issue #52).
    """
    result: dict[tuple[str, tuple[str, ...]], pd.Series] = {}
    for columns, names in groups.items():
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} neutralizer column(s) are not in the features frame "
                f"(first: {missing[:3]}) — load them alongside the models' columns"
            )
        ordered = sorted(names)
        frame = pd.concat([features[list(columns)], source[ordered], era.rename("era")], axis=1)
        projected = era_feature_projection(frame, ordered, list(columns))
        for name in ordered:
            result[(name, columns)] = projected[name]
    return result


def blend_strategies(
    strategies: Mapping[str, Strategy],
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    era: pd.Series,
    feature_sets: FeatureSets,
) -> pd.DataFrame:
    """Each strategy's blended, neutralized prediction: one column per strategy, unranked.

    `predictions` holds one raw prediction column per model, named by
    `ModelSpec.name`; it may hold more models than any one strategy uses, since a
    sweep scores subsets. `features` must contain every column any strategy's
    neutralizations name. All three share one index.

    A strategy neutralizes each model that has a `neutralization`, rank-blends
    the results by `BlendSpec.weights`, then neutralizes the blend if the blend
    has one. Rows of a sweep that share a column and a feature set share one
    projection, however many proportions or weightings they differ by.
    """
    for name, frame in (("features", features), ("era", era)):
        if not frame.index.equals(predictions.index):
            raise ValueError(f"`{name}` must share `predictions`' index — align them before blending")

    used = {spec.name for strategy in strategies.values() for spec in strategy.models}
    missing = sorted(used - set(predictions.columns))
    if missing:
        raise KeyError(f"model(s) {missing} have no predictions (have: {sorted(predictions.columns)})")

    # What each strategy asks for, resolved once: per-model stage, weights, blend stage.
    plans = {}
    for label, strategy in strategies.items():
        models = tuple(
            (spec.name, _resolve(spec.neutralization, feature_sets)) for spec in strategy.models
        )
        weights = strategy.blend.weights
        weights_key = (
            tuple((spec.name, weights[spec.name]) for spec in strategy.models)
            if weights is not None
            else None
        )
        plans[label] = (models, weights_key, _resolve(strategy.blend.neutralization, feature_sets))

    model_groups: dict[tuple[str, ...], set[str]] = {}
    for models, _, _ in plans.values():
        for name, resolved in models:
            if resolved is not None:
                model_groups.setdefault(resolved[1], set()).add(name)
    model_projections = _projections(predictions, model_groups, features, era)

    # One blend per distinct (per-model stage, weighting): the blend-stage
    # proportion is applied afterwards, so every row sharing them shares it.
    blends: dict[tuple, pd.Series] = {}
    for models, weights_key, _ in plans.values():
        key = (models, weights_key)
        if key in blends:
            continue
        per_model = {
            name: predictions[name]
            if resolved is None
            else predictions[name] - resolved[0] * model_projections[(name, resolved[1])]
            for name, resolved in models
        }
        blends[key] = combine_predictions(
            per_model, era, dict(weights_key) if weights_key is not None else None
        )

    slots = {key: f"__blend_{i}" for i, key in enumerate(blends)}
    blend_groups: dict[tuple[str, ...], set[str]] = {}
    for models, weights_key, resolved in plans.values():
        if resolved is not None:
            blend_groups.setdefault(resolved[1], set()).add(slots[(models, weights_key)])
    blend_projections = _projections(
        pd.DataFrame({slots[key]: series for key, series in blends.items()}, index=predictions.index),
        blend_groups,
        features,
        era,
    )

    out = {}
    for label, (models, weights_key, resolved) in plans.items():
        key = (models, weights_key)
        blended = blends[key]
        if resolved is not None:
            proportion, columns = resolved
            blended = blended - proportion * blend_projections[(slots[key], columns)]
        out[label] = blended
    return pd.DataFrame(out, index=predictions.index)
