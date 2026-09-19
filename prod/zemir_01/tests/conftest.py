import pytest

from zemir.strategy import TRAINERS

from factories import fit_predict_column, fit_predict_negated_column


@pytest.fixture
def fake_trainers(monkeypatch):
    """Registers `predict_column`/`predict_negated_column` — trainers that fit nothing.

    `ModelSpec.trainer` resolves through `zemir.strategy.TRAINERS`, so a test
    that wants a model without sklearn/xgboost registers one there rather
    than injecting a callable past the `Strategy`.
    """
    monkeypatch.setitem(TRAINERS, "predict_column", fit_predict_column)
    monkeypatch.setitem(TRAINERS, "predict_negated_column", fit_predict_negated_column)
