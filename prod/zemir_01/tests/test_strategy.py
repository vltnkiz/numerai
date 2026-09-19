from zemir.pipeline import fit_models, predict_each
from zemir.strategy import BlendSpec, ModelSpec, Strategy, build_trainers

from factories import make_dataset


def test_two_era_boosts_at_different_depths_fit_and_predict_as_distinct_models():
    """The expressiveness test #76 exists for.

    Today's `build_trainers` returns a `dict[str, Trainer]` keyed by model
    name, so two `era_boost` specs collide on the key `"era_boost"` and this
    cannot even be expressed. Keying by each `ModelSpec`'s own `name` instead
    is what makes #74's headline claim a red-green fact.
    """
    dataset = make_dataset()
    strategy = Strategy(
        models=(
            ModelSpec(
                name="era_boost_shallow",
                features="medium",
                trainer="xgboost",
                params={
                    "trees_per_step": 5,
                    "num_iters": 0,
                    "max_depth": 1,
                    "colsample_bytree": 1.0,
                },
            ),
            ModelSpec(
                name="era_boost_deep",
                features="medium",
                trainer="xgboost",
                params={
                    "trees_per_step": 5,
                    "num_iters": 0,
                    "max_depth": 4,
                    "colsample_bytree": 1.0,
                },
            ),
        ),
        blend=BlendSpec(),
    )

    train_df = dataset.train.dropna(subset=["target"])
    X, y, era = train_df[dataset.feature_columns], train_df["target"], train_df["era"]

    fitted = fit_models(build_trainers(strategy), X, y, era)
    predictions = predict_each(fitted, dataset.validation, dataset.feature_columns)

    assert set(predictions) == {"era_boost_shallow", "era_boost_deep"}
    assert not predictions["era_boost_shallow"].equals(predictions["era_boost_deep"])


def test_build_trainers_keys_by_model_spec_name_not_trainer_name():
    strategy = Strategy(
        models=(
            ModelSpec(name="a", features="medium", trainer="ols"),
            ModelSpec(name="b", features="medium", trainer="ols"),
        ),
        blend=BlendSpec(),
    )

    trainers = build_trainers(strategy)

    assert set(trainers) == {"a", "b"}
