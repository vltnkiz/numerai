from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
import xgboost as xgb

# (features, target, era) -> a fitted object exposing .predict(features)
Trainer = Callable[[pd.DataFrame, pd.Series, pd.Series], object]


def train_linear(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


@dataclass
class _Float32Predictor:
    """Wraps a fitted sklearn model to keep both fit and predict inputs float32.

    `zemir.data` reads Numerai's features as int8; sklearn's `check_array`
    upcasts anything that isn't already float32/float64 to float64 before
    fitting *or predicting*. At `all` width that float64 design matrix is the
    thing issue #45 found doesn't fit in memory — feeding it int8 or float64
    at predict time (e.g. over the ~4M-row validation set) would reproduce the
    exact same blowup one call later. Casting explicitly at both ends, rather
    than trusting sklearn's dtype-preserving fast path, is what actually keeps
    it float32 throughout.
    """

    model: object

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32))


def train_ridge(X: pd.DataFrame, y: pd.Series, *, alpha: float) -> _Float32Predictor:
    """OLS's regularized fallback for a wider, ill-conditioned feature set (issue #38).

    Plain OLS's normal equations become numerically unstable once the design
    matrix is this collinear; Ridge's L2 penalty conditions the solve without
    changing the model family. Fit in float32, not sklearn's default float64
    upcast — halves the design matrix issue #45 found doesn't fit `all`'s
    3,555 (well, 3,414 post-pruning) features in memory otherwise.

    `copy_X=False` matters at this width: `Ridge.fit` centers `X` before
    solving (`fit_intercept` defaults to `True`), and with the sklearn
    default `copy_X=True` that centering allocates a second full-size float32
    array rather than centering `Xf` in place — issue #47 found this second
    copy, not the cast itself, is what pushed a full `all`-width fit over the
    ~50GB WSL cap (the cast alone peaks under it). `Xf` isn't needed
    uncentered afterward, so fitting in place is free.
    """
    Xf = np.asarray(X, dtype=np.float32)
    yf = np.asarray(y, dtype=np.float32)
    del X, y  # the caller's int8 frame is otherwise held for the rest of this
    # call for no reason — freed before `.fit()` runs, not after, since that's
    # the only way it actually shrinks this call's peak.
    model = Ridge(alpha=alpha, copy_X=False)
    model.fit(Xf, yf)
    return _Float32Predictor(model)


def _scale_features(Xf: np.ndarray) -> np.ndarray:
    """Map int8 feature values (confirmed `{0,1,2,3,4}` for every column) to `[-1, 1]`.

    A fixed, zero-cost affine transform rather than a fitted (running/streaming)
    standardization: every Numerai feature column already shares this exact
    same discrete support, so there is nothing left to learn from a data pass —
    `(x-2)/2` is already centered and unit-scale-ish for every column, with no
    prepass and no state to carry between chunks. In-place on the caller's own
    float32 array, matching `train_ridge`'s no-extra-copy discipline.
    """
    Xf -= 2.0
    Xf /= 2.0
    return Xf


@dataclass
class _ScaledSGDPredictor:
    """Like `_Float32Predictor`, but also applies `train_sgd`'s fixed feature scaling.

    Kept separate from `_Float32Predictor` rather than adding a scaling flag to
    it: `train_sgd` is additive infrastructure (issue #50) that must not touch
    the existing Ridge/OLS predict path.

    Predicts in `chunk_rows`-row slices rather than casting the whole input to
    float32 at once: `_Float32Predictor`'s single-shot cast is exactly what
    `train_sgd` exists to dodge at `all` width, and predicting is no
    exception — casting the full ~4M-row validation set to float32 at 3,414
    columns is ~51GB, the same order of blowup `train_ridge` hits fitting the
    same width (issue #51). Chunked here for the same reason training is.
    """

    model: object
    chunk_rows: int = 200_000

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy(dtype=np.int8, copy=False)
        return np.concatenate(
            [
                self.model.predict(
                    _scale_features(X_arr[start : start + self.chunk_rows].astype(np.float32))
                )
                for start in range(0, len(X_arr), self.chunk_rows)
            ]
        )


def train_sgd(
    X: pd.DataFrame,
    y: pd.Series,
    era: pd.Series,
    *,
    chunk_rows: int = 200_000,
    max_epochs: int = 10,
    patience: int = 2,
    holdout_fraction: float = 0.05,
    random_state: int = 0,
    alpha: float = 0.0001,
    eta0: float = 0.01,
) -> _ScaledSGDPredictor:
    """`all`-width escalation past `train_ridge`'s memory ceiling (issue #50).

    `train_ridge` casts the full int8 `X` to a float32 design matrix in one
    shot — at `all` width that cast alone is ~47GB (issue #49), which is what
    doesn't fit. `SGDRegressor.partial_fit` never needs the full matrix: each
    call below casts and scales only one `chunk_rows`-row slice of the
    still-int8 `X` to float32, fits on it, then frees it, so the transient
    peak is chunk-sized (~2.7GB at the default 200k rows / ~3,414 columns)
    instead of whole-matrix-sized. `X` is converted to a numpy array with
    `copy=False` up front (a pandas-block no-op for a homogeneous-dtype
    frame, not a second resident copy of `X`) purely so chunk slicing is fast
    numpy fancy-indexing rather than repeated `.iloc` calls.

    This trades `train_ridge`'s exact closed-form solution for an
    iterative one — sklearn's own tested `SGDRegressor` implementation,
    not a hand-rolled solve — and needs its own convergence handling:
    `partial_fit` always does exactly one epoch per call and ignores
    `early_stopping`/`n_iter_no_change` (those only apply inside `.fit()`'s
    own loop), so epochs and stopping are managed here instead. Each epoch
    reshuffles the fit rows (a fresh `np.random.default_rng(random_state)`
    permutation, pinned like `train_xgboost`'s `random_state` for
    reproducibility, not because the seed choice itself matters) so the
    model sees the whole training window every epoch rather than drifting
    toward whatever era it saw last (eras are not i.i.d. — issue #50).

    The last `holdout_fraction` of eras (by time, not by row) are held out of
    every `partial_fit` call and used only to score each epoch (plain overall
    Spearman, not the harness's proper per-era `mean_corr` — this is an
    internal stopping signal, not a reported score) — held out by *era*,
    not by row, because adjacent rows within an era are correlated and a
    random-row holdout would leak. Training stops after `patience` epochs in
    a row fail to improve that score.

    `alpha`/`eta0` default to sklearn's own `SGDRegressor` defaults but are
    now exposed for tuning (issue #51): issue #50 found the untuned default
    `alpha` is genuinely unstable at `all` width (regularization needs to
    scale with feature count, same reason `train_ridge` needs `alpha=3e6`
    there instead of a flat value) rather than merely weaker than Ridge.
    `learning_rate`/`power_t` stay at sklearn's defaults — the invscaling
    schedule already decays step size over time, and #51 tunes `alpha` first
    before touching those.
    """
    era_values = era.to_numpy()
    eras_sorted = sorted(pd.unique(era_values), key=int)
    n_holdout = max(1, round(len(eras_sorted) * holdout_fraction))
    holdout_eras = set(eras_sorted[-n_holdout:])
    holdout_mask = pd.Series(era_values).isin(holdout_eras).to_numpy()
    fit_idx = np.flatnonzero(~holdout_mask)

    X_arr = X.to_numpy(dtype=np.int8, copy=False)
    y_arr = y.to_numpy(dtype=np.float32)
    del X, y  # see train_ridge: freed before the loop, not after, to actually shrink peak

    holdout_idx = np.flatnonzero(holdout_mask)
    X_holdout = _scale_features(X_arr[holdout_idx].astype(np.float32))
    y_holdout = y_arr[holdout_idx]

    rng = np.random.default_rng(random_state)
    model = SGDRegressor(random_state=random_state, alpha=alpha, eta0=eta0)

    best_score = -np.inf
    stale_epochs = 0
    for _ in range(max_epochs):
        order = rng.permutation(fit_idx)
        for start in range(0, len(order), chunk_rows):
            chunk = order[start : start + chunk_rows]
            Xb = _scale_features(X_arr[chunk].astype(np.float32))
            model.partial_fit(Xb, y_arr[chunk])

        score = spearmanr(model.predict(X_holdout), y_holdout)[0]
        if score > best_score + 1e-4:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return _ScaledSGDPredictor(model, chunk_rows=chunk_rows)


class _XGBBatchIter(xgb.DataIter):
    """Feeds `QuantileDMatrix` one row-batch at a time instead of one whole matrix.

    XGBoost's own DataFrame path (`xgboost/data.py`'s `pandas_transform_data`)
    converts column by column and appends every column's float32 copy to a list
    it holds all at once. At `all` width that is 3,414 columns x 2,746,268 rows
    = ~37.5GB of float32 on top of `X`'s own int8 — the ceiling issue #51
    measured (RSS climbing smoothly at ~2.5GB/s to a guard kill inside
    `.fit()`), and structurally the identical blowup issue #49 found in
    `train_ridge`, one stage over.

    `DataIter(release_data=True)` is documented for exactly this case ("Set it
    to True if the data transformation (converting data to np.float32 type) is
    memory intensive"): each batch's cast is freed once XGBoost has quantized
    it, so the float32 transient is batch-sized (~2.7GB at 200k rows) rather
    than whole-matrix-sized. What stays resident is the quantized gradient
    index — ~1 byte per entry at the default `max_bin=256`, ~9.4GB — not a
    float32 design matrix.

    `rows` selects a subset positionally and is applied *per batch*, not up
    front: the era-boost loop's worst-`proportion` refit would otherwise
    materialize a 4.7GB int8 copy of half of `X` before the iterator even
    starts, and hold it alongside the matrix being built from it.

    Batching does not change the model here. Numerai's features are int8 with
    exact support `{0,1,2,3,4}` (confirmed in issue #50), and `max_bin=256` is
    far past 5 distinct values, so the quantile sketch is exact however the
    rows are split — verified bit-identical against a single-batch build, and
    against `XGBRegressor.fit` itself, before this was written (issue #52).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        batch_rows: int,
        rows: np.ndarray | None = None,
    ) -> None:
        self._X = X
        self._y = y
        self._batch_rows = batch_rows
        self._rows = rows
        self._n = len(X) if rows is None else len(rows)
        self._start = 0
        super().__init__(release_data=True)

    def reset(self) -> None:
        self._start = 0

    def next(self, input_data: Callable) -> bool:
        if self._start >= self._n:
            return False
        stop = self._start + self._batch_rows
        take = slice(self._start, stop) if self._rows is None else self._rows[self._start : stop]
        input_data(data=self._X[take].astype(np.float32), label=self._y[take])
        self._start = stop
        return True


@dataclass
class _ChunkedBoosterPredictor:
    """A fitted booster that predicts in row-chunks, off the still-int8 matrix.

    The counterpart to `_ScaledSGDPredictor` for the tree stage, and needed for
    the same reason: `predict_each` calls this over the ~4M-row validation set,
    which at `all` width is ~54GB as a single float32 conversion — a larger
    blowup than training's. `inplace_predict` accepts the int8 array directly
    (measured bit-identical to passing float32), so a chunk here costs only
    XGBoost's own internal per-chunk handling, and no explicit cast is made.

    Wraps the raw `Booster` rather than exposing it, so `train_xgboost` keeps
    returning something whose only interface is `.predict(features)` — what
    `zemir.pipeline`'s `predict_each` and the `Trainer` contract expect, and
    what `XGBRegressor` used to provide.
    """

    booster: xgb.Booster
    chunk_rows: int = 200_000

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X if isinstance(X, np.ndarray) else X.to_numpy(dtype=np.int8, copy=False)
        return np.concatenate(
            [
                self.booster.inplace_predict(X_arr[start : start + self.chunk_rows])
                for start in range(0, len(X_arr), self.chunk_rows)
            ]
        )


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    era: pd.Series,
    *,
    trees_per_step: int = 50,
    num_iters: int = 40,
    proportion: float = 0.5,
    learning_rate: float = 0.01,
    max_depth: int = 5,
    colsample_bytree: float = 0.1,
    random_state: int = 0,
    batch_rows: int = 200_000,
    on_iteration: Callable[[int, "_ChunkedBoosterPredictor", list], None] | None = None,
) -> _ChunkedBoosterPredictor:
    """Fit, then repeatedly re-fit on the worst-scoring `proportion` of eras.

    The 40-iteration, 2050-tree budget and `proportion=0.5` are the canonical
    additive-boosting loop run ~10x its original budget (issue #27) — measured
    to beat a plain single fit (`num_iters=0`) on mean_corr/sharpe/smart_sharpe,
    though it costs mean_mmc (issue #31). Whether 2050 trees is past the useful
    point on that curve, and whether the worst-era set locks onto a fixed
    subset instead of churning, is still unverified — flagged in issue #31 for
    a follow-up rather than blocking this default.

    Built on XGBoost's native `xgb.train` rather than the `XGBRegressor`
    sklearn wrapper (issue #52), because the wrapper takes no `DMatrix` and so
    cannot be handed a streamed one — and streaming is what gets `all` width
    under this machine's memory ceiling (see `_XGBBatchIter`). The two are
    otherwise the same fit: `XGBRegressor.fit` builds a single-batch
    `QuantileDMatrix` internally, and was verified to produce bit-identical
    predictions to this path before the switch. `nthread`/`seed` are the native
    spellings of the wrapper's `n_jobs=-1`/`random_state`.

    All three call sites into XGBoost move together rather than only the two
    that provably blow up: `QuantileDMatrix` cannot be `.slice()`d ("Slicing
    DMatrix is not supported for external memory"), so the loop's worst-era
    refit has to build its own matrix regardless, and a partial rewrite would
    leave two dialects in one function for no saving.

    `ref=dtrain` pins every refit to the initial matrix's cut points. The cuts
    are exact over the features' 5 distinct values and so almost never differ,
    but a worst-era slice that happens never to observe some value in some
    column would otherwise quantize that column differently — a silent
    per-iteration drift, and the loop runs 40 of them.

    `random_state` is explicit only to document intent: xgboost already falls
    back to seed 0 internally, so fits were already bit-identical (issue #27).

    `on_iteration`, when given, is called once before the loop (iteration 0,
    the plain-fit anchor, with an empty worst-era list) and once after each of
    the `num_iters` refits, with that refit's own worst-era list — an
    observability seam for issue #36's budget-curve probe, never exercised by
    the live path (`None` by default, and the loop's own memory shape is
    unchanged either way).
    """
    params = {
        "objective": "reg:squarederror",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "nthread": -1,
        "seed": random_state,
    }

    # See train_ridge/train_sgd: `copy=False` is a pandas-block no-op for a
    # homogeneous int8 frame, not a second resident copy, and the caller's
    # bindings are dropped before anything large is allocated.
    X_arr = X.to_numpy(dtype=np.int8, copy=False)
    y_arr = y.to_numpy(dtype=np.float32)
    era_values = era.to_numpy()
    del X, y, era

    dtrain = xgb.QuantileDMatrix(_XGBBatchIter(X_arr, y_arr, batch_rows=batch_rows))
    booster = xgb.train(params, dtrain, num_boost_round=trees_per_step)
    predictor = _ChunkedBoosterPredictor(booster, chunk_rows=batch_rows)

    # `meta` carries only era/target/pred, not X's (up to 3,555-column) feature
    # data — issue #47 found the old `working = X.copy()` held a second
    # full-width copy of X resident for all `num_iters` iterations on top of X
    # itself. The worst-era rows are now carried as positional indices, handed
    # to `_XGBBatchIter` to slice batch by batch, so not even one iteration's
    # subset is materialized whole.
    meta = pd.DataFrame({"era": era_values, "target": y_arr})

    if on_iteration is not None:
        on_iteration(0, predictor, [])

    for iteration in range(1, num_iters + 1):
        meta["pred"] = predictor.predict(X_arr)
        era_scores = (
            meta.groupby("era")
            .apply(lambda d: spearmanr(d["pred"], d["target"])[0])
            .to_dict()
        )
        worst_eras = sorted(era_scores, key=era_scores.get)[
            : int(len(era_scores) * proportion)
        ]
        worst_rows = np.flatnonzero(meta["era"].isin(worst_eras).to_numpy())

        # Dropped before the next one is built, not merely rebound after:
        # `dworst = xgb.QuantileDMatrix(...)` evaluates its right-hand side
        # while the previous iteration's matrix is still bound, holding two
        # ~4.4GiB quantized indices at once for no reason, on every one of the
        # `num_iters` iterations (issue #52).
        dworst = None

        # xgb_model= adds `num_boost_round` NEW trees on top of the existing
        # booster, not a target total, so it stays pinned at trees_per_step.
        dworst = xgb.QuantileDMatrix(
            _XGBBatchIter(X_arr, y_arr, batch_rows=batch_rows, rows=worst_rows),
            ref=dtrain,
        )
        booster = xgb.train(
            params, dworst, num_boost_round=trees_per_step, xgb_model=booster
        )
        predictor = _ChunkedBoosterPredictor(booster, chunk_rows=batch_rows)

        if on_iteration is not None:
            on_iteration(iteration, predictor, worst_eras)

    return predictor
