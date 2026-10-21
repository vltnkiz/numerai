import pytest
from numerapi import NumerAPI

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


@pytest.fixture(autouse=True)
def no_dataset_downloads(monkeypatch):
    """Fails any test that reaches Numerai's real dataset download.

    A read nobody stubbed would otherwise fetch gigabytes into `datasets/` and
    hang the suite. Tests that exercise downloading hand in their own fake
    `napi`, which this does not touch.
    """

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"a test reached NumerAPI.download_dataset{args}: stub the read")

    monkeypatch.setattr(NumerAPI, "download_dataset", refuse)
