#!/usr/bin/env python3
"""Fetch Numerai's per-round scores for every submission slot, update live_scores.jsonl, print the headline.

The same step the scheduled live run takes after its submission, runnable any
time without training or submitting. Reads `NUMERAI_MODELS` from `.env`; the
scores themselves are public, so no key is used.

Usage:
  python scripts/live_scores.py
"""

from __future__ import annotations

from dotenv import load_dotenv

from zemir.live_scores import LIVE_SCORES_PATH, format_resolved, summarize_resolved, update_record
from zemir.pipeline import load_numerai_models


def main() -> None:
    load_dotenv()
    rows = update_record(load_numerai_models())
    print(format_resolved(summarize_resolved(rows)))
    print(f"\n{len(rows)} rounds recorded in {LIVE_SCORES_PATH}")


if __name__ == "__main__":
    main()
