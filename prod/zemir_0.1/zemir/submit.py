"""Submission stage: upload neutralized predictions directly to Numerai.

Submits to exactly one named slot in `NUMERAI_MODELS` (zemir_0.1/.env,
extended over time via scripts/add_numerai_model.sh) — per [Ensembling
strategy](https://github.com/vltnkiz/numerai/issues/14) and [Prediction-to-
slot routing](https://github.com/vltnkiz/numerai/issues/15), one run produces
one combined prediction series for one Numerai model slot, not the same
predictions fanned out to every slot `NUMERAI_MODELS` happens to hold. Per
docs/research/live-round-data-and-submission.md: `upload_predictions` accepts
a DataFrame directly (serialized via `to_csv(index=False)`, so `id` must be a
column, not the index, or it's silently dropped) and has no `round_num`
param — the round is implicit in whatever round is currently open.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI


@dataclass
class SubmissionResult:
    model_name: str
    model_id: str
    submission_id: str


def load_numerai_models(env: dict[str, str] | None = None) -> dict[str, str]:
    """Parse `NUMERAI_MODELS` (comma-separated `name=model_id` pairs) into a dict."""
    env = env if env is not None else os.environ
    raw = env.get("NUMERAI_MODELS", "")
    if not raw.strip():
        raise ValueError("NUMERAI_MODELS is empty — add a model slot first")

    models: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, model_id = pair.partition("=")
        if not name or not model_id:
            raise ValueError(f"malformed NUMERAI_MODELS entry: {pair!r}")
        models[name] = model_id
    return models


def _submission_frame(
    predictions: pd.Series, *, id_col: str = "id", prediction_col: str = "prediction"
) -> pd.DataFrame:
    frame = predictions.rename(prediction_col).reset_index()
    frame = frame.rename(columns={frame.columns[0]: id_col})
    return frame[[id_col, prediction_col]]


def submit_predictions(
    predictions: pd.Series,
    *,
    model_slot: str,
    napi: NumerAPI | None = None,
    models: dict[str, str] | None = None,
) -> SubmissionResult:
    """Upload `predictions` (indexed by `id`) to the `model_slot` slot in `NUMERAI_MODELS`.

    `predictions` is expected to already be neutralized live-round output —
    one combined series for this run, submitted to exactly the one Numerai
    model slot `model_slot` names. `models` holding other slots (e.g. future
    models run in parallel for comparison) are left untouched; picking which
    slot a given run targets is `model_slot`'s job, not a fan-out here.
    """
    load_dotenv()
    napi = napi or NumerAPI(
        public_id=os.environ["NUMERAI_PUBLIC_ID"],
        secret_key=os.environ["NUMERAI_SECRET_KEY"],
    )
    models = models if models is not None else load_numerai_models()
    if model_slot not in models:
        raise ValueError(
            f"model slot {model_slot!r} not found in NUMERAI_MODELS "
            f"(have: {sorted(models)})"
        )

    frame = _submission_frame(predictions)
    model_id = models[model_slot]
    submission_id = napi.upload_predictions(df=frame, model_id=model_id)
    return SubmissionResult(model_name=model_slot, model_id=model_id, submission_id=submission_id)
