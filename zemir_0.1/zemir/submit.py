"""Submission stage: upload neutralized predictions directly to Numerai.

Loops over every `name=model_id` pair in `NUMERAI_MODELS` (zemir_0.1/.env,
extended over time via scripts/add_numerai_model.sh) rather than hardcoding a
single model_id, submitting the same predictions to each in turn. Per
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
    napi: NumerAPI | None = None,
    models: dict[str, str] | None = None,
) -> list[SubmissionResult]:
    """Upload `predictions` (indexed by `id`) to every model in `NUMERAI_MODELS`.

    `predictions` is expected to already be neutralized live-round output.
    Submits the identical frame to each model slot — Numerai scores each
    model independently, so running several models "in parallel for
    comparison" means each gets the same round's predictions from whichever
    model produced them, not a per-model prediction set.
    """
    load_dotenv()
    napi = napi or NumerAPI(
        public_id=os.environ["NUMERAI_PUBLIC_ID"],
        secret_key=os.environ["NUMERAI_SECRET_KEY"],
    )
    models = models if models is not None else load_numerai_models()

    frame = _submission_frame(predictions)

    results = []
    for name, model_id in models.items():
        submission_id = napi.upload_predictions(df=frame, model_id=model_id)
        results.append(
            SubmissionResult(model_name=name, model_id=model_id, submission_id=submission_id)
        )
    return results
