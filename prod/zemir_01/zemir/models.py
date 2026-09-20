from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
import xgboost as xgb


@dataclass(frozen=True)
class FittedModel:
    """A trained estimator, bound to the columns it was fitted on and to how it is fed.

    Predicting on the wrong columns is silent — a tree stage indexes by
    position — so the columns travel with the model, and `predict` takes a
    frame of *any* width that contains them. Nothing here remembers which
    `ModelSpec` produced it (see CONTEXT.md, "Fitted model").

    Everything that differs between the trainers at predict time is *data* on
    this one class, so there is one predict path to get right rather than one
    adapter per trainer:

    - `dtype` is what the estimator is fed. `zemir.data` reads Numerai's
      features as int8, and sklearn's `check_array` upcasts anything that
      isn't already float32/float64 to float64 before fitting *or predicting*.
      At `all` width that float64 design matrix is the thing issue #45 found
      doesn't fit in memory, and feeding it int8 at predict time (over the
      ~4M-row validation set) would reproduce the same blowup one call later.
      Casting explicitly, rather than trusting sklearn's implicit upcast, is
      what makes the width of the cast a decision instead of an accident. It
      is a value the trainer records at fit time, not a property of a class
      chosen by trainer name, so a model at another precision is a different
      `dtype=`, not a different type.
    - `order` is the memory layout of that cast: `"K"` keeps the input's own
      (a frame's block is column-major), `"C"` forces row-major. It is data
      because it changes the *bits*, not just speed — BLAS sums a column-major
      and a row-major matrix in a different order — so a model's numbers are
      only reproduced by feeding it the layout they were established with.
    - `chunk_rows`, when set, predicts in row slices rather than casting the
      whole input at once: the single-shot cast over the ~4M-row validation set
      is ~51GB at `all` width for a float32 linear model and ~54GB for the
      booster (issue #51). `None` predicts in one shot.
    - `transform`, when set, is applied in place to each cast slice before the
      estimator sees it — the *same* function the trainer applied while
      fitting, because the trainer trains through this object (see
      `train_sgd`) rather than keeping its own copy of the rule.

    A booster is predicted with `inplace_predict`, which takes the still-int8
    array directly (measured bit-identical to passing float32), so its `dtype`
    is int8 and a slice costs only XGBoost's own per-slice handling.

    `estimator` is the one named door to the trained object (`explain`, and
    tests reaching for `.coef_`). It may be anything exposing `.predict` —
    a test double needs no sklearn.
    """

    estimator: object
    columns: tuple[str, ...]
    dtype: npt.DTypeLike
    order: Literal["K", "C"] = "K"
    chunk_rows: int | None = None
    transform: Callable[[np.ndarray], np.ndarray] | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_array(df[list(self.columns)].to_numpy(copy=False))

    def explain(self) -> pd.DataFrame:
        """What this fit leans on: one row per fitted column, in the estimator's own quantities.

        A linear model gives `coef` (signed, in the space the estimator was
        fitted in — `sgd` is not rescaled, so it reads 2x what OLS would for
        the same effect); a booster gives `weight` (splits on the feature),
        `gain` (mean loss reduction per split) and `cover` (mean hessian mass
        per split). A booster feature it never split on has `weight` 0 and
        NaN `gain`/`cover` — a mean over zero splits is undefined. The
        `total_*` importances are left out: `total_gain = weight * gain`, and
        they are the worst-reproducing of the five (issue #83).

        **This describes this fit exactly. It is not an estimate of what
        matters in the data**: at production settings a booster's ranking
        barely reproduces across seeds and a linear model's does not
        reproduce across time (issue #83), and the columns are not comparable
        across trainers. An estimator that can do neither raises rather than
        returning an empty frame, which would read as "the model uses nothing".
        """
        index = pd.Index(self.columns, name="feature")
        if isinstance(self.estimator, xgb.Booster):
            return pd.DataFrame(
                {
                    "weight": self._booster_score("weight", missing=0.0),
                    "gain": self._booster_score("gain", missing=np.nan),
                    "cover": self._booster_score("cover", missing=np.nan),
                },
                index=index,
            )
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            raise NotImplementedError(
                f"{type(self.estimator).__name__} exposes neither a booster nor `coef_`, "
                "so there is nothing for explain() to report"
            )
        return pd.DataFrame({"coef": np.asarray(coef, dtype=np.float64)}, index=index)

    def save(self, directory: Path) -> None:
        """Write this model into `directory`, in formats that outlive a library upgrade.

        Not a pickle: an sklearn estimator's pickle is only readable by the
        version that wrote it, and a cache is meant to be read weeks later.
        A booster goes to XGBoost's own `.ubj` model format; a linear model
        (`LinearRegression`, `Ridge`, `SGDRegressor` — the only sklearn
        estimators the trainers produce) is reduced to what its `predict` and
        `explain` read, `coef_` and `intercept_`, as an `.npz`. Its
        hyperparameters are *not* kept — a reloaded estimator predicts and
        explains, it is not a model to refit.

        `meta.json` carries everything else `predict` needs (the columns and
        how the estimator is fed) and is written last, so a directory without
        it is an interrupted save and `load` refuses it. Anything that cannot
        be written faithfully — another estimator, a `transform` that is not
        registered — raises before a byte is written, rather than producing a
        model that loads and then predicts differently.
        """
        transform = None
        if self.transform is not None:
            transform = next((n for n, fn in _TRANSFORMS.items() if fn is self.transform), None)
            if transform is None:
                raise TypeError(f"transform {self.transform!r} is not registered, so it cannot be saved")

        meta = {
            "format": _FORMAT_VERSION,
            "columns": list(self.columns),
            "dtype": np.dtype(self.dtype).name,
            "order": self.order,
            "chunk_rows": self.chunk_rows,
            "transform": transform,
        }
        estimator = self.estimator
        if isinstance(estimator, xgb.Booster):
            meta["estimator"] = _BOOSTER
        elif type(estimator) in _LINEAR_ESTIMATORS.values():
            meta["estimator"] = type(estimator).__name__
            coef, intercept = np.asarray(estimator.coef_), np.asarray(estimator.intercept_)
        else:
            raise TypeError(f"{type(estimator).__name__} is not an estimator this can save")

        directory.mkdir(parents=True, exist_ok=True)
        if isinstance(estimator, xgb.Booster):
            estimator.save_model(str(directory / _BOOSTER_FILE))
        else:
            np.savez(directory / _LINEAR_FILE, coef=coef, intercept=intercept)
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> FittedModel:
        """The model `save` wrote into `directory`; raises rather than guessing if it cannot.

        There is no fallback of any kind — a directory that is missing,
        interrupted, from a newer format or naming an estimator this does not
        know is an error naming what is wrong, never an empty or refitted model.
        """
        meta = json.loads((directory / _META_FILE).read_text())
        if meta.get("format") != _FORMAT_VERSION:
            raise ValueError(
                f"{directory} is format {meta.get('format')!r}; this reads format {_FORMAT_VERSION}"
            )
        columns = tuple(meta["columns"])
        kind = meta["estimator"]
        if kind == _BOOSTER:
            estimator: object = xgb.Booster()
            estimator.load_model(str(directory / _BOOSTER_FILE))
        elif kind in _LINEAR_ESTIMATORS:
            with np.load(directory / _LINEAR_FILE) as arrays:
                coef, intercept = arrays["coef"], arrays["intercept"]
            if coef.shape != (len(columns),):
                raise ValueError(
                    f"{directory} has {coef.shape} coefficients for {len(columns)} columns"
                )
            estimator = _LINEAR_ESTIMATORS[kind]()
            estimator.coef_, estimator.intercept_ = coef, intercept
            estimator.n_features_in_ = len(columns)
        else:
            raise ValueError(f"{directory} names an estimator this does not know: {kind!r}")

        transform = meta["transform"]
        if transform is not None and transform not in _TRANSFORMS:
            raise ValueError(f"{directory} names a transform this does not know: {transform!r}")
        return cls(
            estimator,
            columns,
            dtype=np.dtype(meta["dtype"]),
            order=meta["order"],
            chunk_rows=meta["chunk_rows"],
            transform=None if transform is None else _TRANSFORMS[transform],
        )

    def _booster_score(self, importance_type: str, *, missing: float) -> np.ndarray:
        """`get_score` keyed back onto `columns`.

        `train_xgboost` builds its matrix from a numpy array, so the booster
        knows its features only as `f0..fN`; `columns` is what names them.
        Every key must be one of those, so a booster that was given real names
        (and would key by them) fails here rather than silently reading zeros.
        """
        values = np.full(len(self.columns), missing, dtype=np.float64)
        for key, score in self.estimator.get_score(importance_type=importance_type).items():
            match = re.fullmatch(r"f(\d+)", key)
            if match is None or int(match[1]) >= len(self.columns):
                raise ValueError(
                    f"booster feature {key!r} is not a position within its {len(self.columns)} "
                    "fitted columns"
                )
            values[int(match[1])] = score
        return values

    def _prepare(self, rows: np.ndarray) -> np.ndarray:
        """`rows` as the estimator is fed: cast to `dtype`, then `transform`ed.

        Copies only when it has to. A `transform` works in place, so it must
        never be handed the caller's own array (`copy=True` even if the dtype
        already matches); with no transform, a matching dtype is passed
        through as a view.
        """
        prepared = rows.astype(self.dtype, order=self.order, copy=self.transform is not None)
        return prepared if self.transform is None else self.transform(prepared)

    def _predict_array(self, rows: np.ndarray) -> np.ndarray:
        """The one chunked predict. `predict` is column-checked on top of it;
        `train_xgboost`'s refit loop, which holds only the array, calls it directly."""
        if self.chunk_rows is None:
            return self._predict_prepared(self._prepare(rows))
        return np.concatenate(
            [
                self._predict_prepared(self._prepare(rows[start : start + self.chunk_rows]))
                for start in range(0, len(rows), self.chunk_rows)
            ]
        )

    def _predict_prepared(self, prepared: np.ndarray) -> np.ndarray:
        if isinstance(self.estimator, xgb.Booster):
            return self.estimator.inplace_predict(prepared)
        return self.estimator.predict(prepared)


# What `FittedModel.save` writes, and the only things `load` will accept back.
_FORMAT_VERSION = 1
_META_FILE, _BOOSTER_FILE, _LINEAR_FILE = "meta.json", "booster.ubj", "linear.npz"
_BOOSTER = "xgboost.Booster"
_LINEAR_ESTIMATORS = {cls.__name__: cls for cls in (LinearRegression, Ridge, SGDRegressor)}


# (features, target, era) -> a FittedModel bound to the columns of `features`
Trainer = Callable[[pd.DataFrame, pd.Series, pd.Series], FittedModel]


def train_linear(X: pd.DataFrame, y: pd.Series) -> FittedModel:
    """Plain OLS, fitted and predicted at float64.

    Fitted on the frame's own int8 array, not the frame: sklearn upcasts it to
    float64 either way (the same array, so the same coefficients bit for bit),
    but a frame would also stamp `feature_names_in_` on the estimator, and
    `FittedModel` predicts from an array — sklearn would then warn on every
    call, with `columns` already carrying the names. Deliberately *not*
    pre-cast to float64 here: `LinearRegression`'s default `copy_X=True` would
    then copy that array a second time (it shares memory with its input), where
    the implicit upcast it does itself is the only copy.

    Predicting is where this used to be implicit. `LinearRegression.predict`
    does not upcast — it hands the int8 frame straight to `X @ coef_`, and
    numpy's own int8 -> float64 promotion inside that matmul makes a full
    row-major float64 copy (the ~34GB spike issue #80 measured over the
    validation set). The cast is now `FittedModel`'s, and explicit, at the same
    width — and row-major (`order="C"`), because that is the layout those
    numbers were produced with: a column-major float64 cast, which is what a
    plain `astype` of a frame's block gives, sums differently and moves the last
    bits. Not chunked (`chunk_rows` stays `None`), for the same reason as before:
    chunking would change the GEMM shapes.
    """
    columns = tuple(X.columns)
    model = LinearRegression()
    model.fit(X.to_numpy(copy=False), y)
    return FittedModel(model, columns, dtype=np.float64, order="C")


def train_ridge(X: pd.DataFrame, y: pd.Series, *, alpha: float) -> FittedModel:
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
    columns = tuple(X.columns)
    Xf = np.asarray(X, dtype=np.float32)
    yf = np.asarray(y, dtype=np.float32)
    del X, y  # the caller's int8 frame is otherwise held for the rest of this
    # call for no reason — freed before `.fit()` runs, not after, since that's
    # the only way it actually shrinks this call's peak.
    model = Ridge(alpha=alpha, copy_X=False)
    model.fit(Xf, yf)
    return FittedModel(model, columns, dtype=np.float32)


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


# A `transform` is a function, and a function cannot be written to disk, so
# `FittedModel.save` stores its name here and `load` looks it back up. A new
# transform has to be registered to be persisted. A trainer picks its own
# transform; `ModelSpec` has no field for one (issue #90).
_TRANSFORMS = {"scale_features": _scale_features}


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
) -> FittedModel:
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

    The `FittedModel` is built around the still-unfitted `SGDRegressor` and
    every fit-time slice goes through its `_prepare`, the very method
    `predict` uses. The feature scaling is therefore applied identically at
    fit and predict *by construction*, not by two call sites that happen to
    agree — before this, `_scale_features` was applied here and re-applied by
    a separate predictor class, with nothing enforcing that they matched.

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

    columns = tuple(X.columns)
    X_arr = X.to_numpy(dtype=np.int8, copy=False)
    y_arr = y.to_numpy(dtype=np.float32)
    del X, y  # see train_ridge: freed before the loop, not after, to actually shrink peak

    rng = np.random.default_rng(random_state)
    model = SGDRegressor(random_state=random_state, alpha=alpha, eta0=eta0)
    fitted = FittedModel(
        model, columns, dtype=np.float32, chunk_rows=chunk_rows, transform=_scale_features
    )

    holdout_idx = np.flatnonzero(holdout_mask)
    X_holdout = fitted._prepare(X_arr[holdout_idx])
    y_holdout = y_arr[holdout_idx]

    best_score = -np.inf
    stale_epochs = 0
    for _ in range(max_epochs):
        order = rng.permutation(fit_idx)
        for start in range(0, len(order), chunk_rows):
            chunk = order[start : start + chunk_rows]
            model.partial_fit(fitted._prepare(X_arr[chunk]), y_arr[chunk])

        score = spearmanr(model.predict(X_holdout), y_holdout)[0]
        if score > best_score + 1e-4:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return fitted


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
    on_iteration: Callable[[int, FittedModel, list], None] | None = None,
) -> FittedModel:
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
    columns = tuple(X.columns)
    X_arr = X.to_numpy(dtype=np.int8, copy=False)
    y_arr = y.to_numpy(dtype=np.float32)
    era_values = era.to_numpy()
    del X, y, era

    dtrain = xgb.QuantileDMatrix(_XGBBatchIter(X_arr, y_arr, batch_rows=batch_rows))
    booster = xgb.train(params, dtrain, num_boost_round=trees_per_step)
    predictor = FittedModel(booster, columns, dtype=np.int8, chunk_rows=batch_rows)

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
        meta["pred"] = predictor._predict_array(X_arr)
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
        predictor = FittedModel(booster, columns, dtype=np.int8, chunk_rows=batch_rows)

        if on_iteration is not None:
            on_iteration(iteration, predictor, worst_eras)

    return predictor
