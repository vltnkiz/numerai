"""`zemir.fitting`: per-model feature slices under the memory discipline (issue #80).

The discipline is about *when* things are allocated, held and released, so
these tests observe timing — weakrefs to the frames a trainer is handed,
and an event log interleaving pyarrow pool releases with fits — rather than
measuring bytes. The bytes are measured once, at full scale, on the real fit.
"""

import gc
import weakref
from dataclasses import replace

import pandas as pd
import pytest

from zemir import fitting
from zemir.fitting import fit_strategy
from zemir.pipeline import predict_each
from zemir.strategy import TRAINERS, BlendSpec, ModelSpec, Strategy

from factories import FEATURE_COLUMNS, FEATURE_SETS, PredictColumn, make_dataset


class Recorder:
    """A registered `record` trainer that notes what each fit was handed, and when."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.columns: dict[str, tuple[str, ...]] = {}
        self.frame_ids: dict[str, int] = {}
        self.refs: dict[str, weakref.ref] = {}
        self.dead_at_fit: dict[str, dict[str, bool]] = {}

    def __call__(self, X, y, era, *, tag: str, column: str) -> PredictColumn:
        self.events.append(f"fit:{tag}")
        self.columns[tag] = tuple(X.columns)
        self.frame_ids[tag] = id(X)
        self.dead_at_fit[tag] = {t: ref() is None for t, ref in self.refs.items()}
        self.refs[tag] = weakref.ref(X)
        return PredictColumn(column)


@pytest.fixture
def recorder(monkeypatch):
    recorder = Recorder()
    monkeypatch.setitem(TRAINERS, "record", recorder)

    class PoolSpy:
        def release_unused(self) -> None:
            recorder.events.append("release")

    monkeypatch.setattr(fitting.pa, "default_memory_pool", lambda: PoolSpy())
    return recorder


def spec(tag: str, features: str, column: str = "feature_a") -> ModelSpec:
    return ModelSpec(
        name=tag, features=features, trainer="record", params={"tag": tag, "column": column}
    )


def strategy_of(*specs: ModelSpec) -> Strategy:
    return Strategy(models=specs, blend=BlendSpec())


def fit(strategy: Strategy, dataset, **kwargs):
    return fit_strategy(strategy, lambda: dataset.train, feature_sets=FEATURE_SETS, **kwargs)


def test_two_specs_at_different_widths_fit_and_predict_on_their_own_columns(recorder):
    dataset = make_dataset()
    strategy = strategy_of(
        spec("narrow", "narrow", column="feature_a"),
        spec("wide", "medium", column="feature_c"),
    )

    result = fit(strategy, dataset)

    assert recorder.columns["narrow"] == ("feature_a",)
    assert recorder.columns["wide"] == tuple(FEATURE_COLUMNS)

    predictions = predict_each(result.models, dataset.validation)
    assert predictions["narrow"].tolist() == dataset.validation["feature_a"].tolist()
    assert predictions["wide"].tolist() == dataset.validation["feature_c"].tolist()
    # Predicting hands each model exactly the columns it was fitted on, from a
    # frame that has far more of them.
    assert result.models["narrow"].model.predicted_on == ["feature_a"]
    assert result.models["wide"].model.predicted_on == FEATURE_COLUMNS


def test_a_spec_naming_the_whole_union_is_handed_the_union_frame_itself(recorder):
    # The shipped strategies' whole memory story: every model names one
    # feature set, so one full-width matrix is live — never a copy per model.
    dataset = make_dataset()

    fit(strategy_of(spec("a", "medium"), spec("b", "medium")), dataset)

    assert recorder.frame_ids["a"] == recorder.frame_ids["b"]


def test_a_narrower_slice_is_freed_as_soon_as_its_model_is_fitted(recorder):
    dataset = make_dataset()

    fit(strategy_of(spec("narrow", "narrow"), spec("wide", "medium")), dataset)

    # By the time the wide model is fitting, the narrow slice is already gone.
    assert recorder.dead_at_fit["wide"] == {"narrow": True}


def test_no_frame_outlives_the_fit(recorder):
    dataset = make_dataset()

    fit(strategy_of(spec("narrow", "narrow"), spec("wide", "medium")), dataset)

    assert {tag: ref() for tag, ref in recorder.refs.items()} == {"narrow": None, "wide": None}


def test_the_wide_train_frame_is_gone_before_any_trainer_runs(monkeypatch):
    # Issue #47: a `train_df` still bound while the trainers run is a second
    # full-width copy live beside the one being fitted. Counted by looking for
    # frames still carrying `target` at the train split's own length — the
    # caller's `dataset.train` is the one legitimate survivor.
    dataset = make_dataset(n_train_eras=10, n_validation_eras=5, rows_per_era=20)
    seen: list[int] = []

    def counting_trainer(X, y, era, *, column):
        seen.append(
            sum(
                1
                for o in gc.get_objects()
                if isinstance(o, pd.DataFrame) and "target" in o.columns and len(o) == len(dataset.train)
            )
        )
        return PredictColumn(column)

    monkeypatch.setitem(TRAINERS, "counting", counting_trainer)
    counting = ModelSpec(name="m", features="medium", trainer="counting", params={"column": "feature_a"})

    fit_strategy(
        strategy_of(counting), lambda: dataset.train.copy(), feature_sets=FEATURE_SETS
    )

    assert seen == [1]


def test_pool_is_released_before_each_trainer_and_once_after_the_last(recorder):
    dataset = make_dataset()

    fit(strategy_of(spec("a", "medium"), spec("b", "medium")), dataset)

    assert recorder.events == ["release", "fit:a", "release", "fit:b", "release"]


def test_narrowest_fits_first_but_models_come_back_in_declared_order(recorder):
    # The widest model runs last, with the union freed beside it; nothing about
    # the result depends on the order, so the declared one is what's returned.
    dataset = make_dataset()

    result = fit(strategy_of(spec("wide", "medium"), spec("narrow", "narrow")), dataset)

    assert [e for e in recorder.events if e.startswith("fit:")] == ["fit:narrow", "fit:wide"]
    assert list(result.models) == ["wide", "narrow"]


def test_equal_width_specs_fit_in_declared_order(recorder):
    dataset = make_dataset()

    fit(strategy_of(spec("first", "medium"), spec("second", "medium")), dataset)

    assert [e for e in recorder.events if e.startswith("fit:")] == ["fit:first", "fit:second"]


def test_only_target_bearing_rows_of_the_last_eras_are_fitted(recorder):
    dataset = make_dataset(n_train_eras=10, rows_per_era=20)
    dataset.train.loc[dataset.train.index[:5], "target"] = None

    result = fit(strategy_of(spec("a", "medium")), dataset, max_eras=3)

    assert result.train_eras == 3


def test_a_model_naming_an_unresolved_feature_set_fails_loudly(recorder):
    dataset = make_dataset()

    with pytest.raises(KeyError, match="'huge'"):
        fit(strategy_of(spec("a", "huge")), dataset)


def test_fit_order_does_not_change_what_a_model_learns():
    # Real trainers, not fakes: OLS on the narrow slice and on the union frame
    # must each learn from exactly their own columns.
    dataset = make_dataset()
    strategy = Strategy(
        models=(
            ModelSpec(name="wide", features="medium", trainer="ols"),
            ModelSpec(name="narrow", features="narrow", trainer="ols"),
        ),
        blend=BlendSpec(),
    )
    reversed_strategy = replace(strategy, models=tuple(reversed(strategy.models)))

    a = fit(strategy, dataset).models
    b = fit(reversed_strategy, dataset).models

    for name in ("wide", "narrow"):
        assert (
            a[name].predict(dataset.validation).tolist()
            == b[name].predict(dataset.validation).tolist()
        )
    assert a["narrow"].model.coef_.shape == (1,)
    assert a["wide"].model.coef_.shape == (3,)
