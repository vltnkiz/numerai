# zemir_01

Numerai Classic pipeline: download → train → combine → neutralize → rank-normalize → submit.

## Two entrypoints — only one of them submits

| Command | Submits? | Artifacts |
| --- | --- | --- |
| `python scripts/run_pipeline.py --model linear` | **YES — uploads to the live `zemir_01` slot** | `runs/<run_id>/` |
| `python scripts/run_experiment.py --model ensemble` | No | `runs/experiments/<run_id>-<model>/` |
| `python scripts/run_experiment.py --model ensemble --smoke` | No | `runs/experiments/<run_id>-<model>-smoke/` |

Submission is not a flag on the pipeline — it is a separate function.
`zemir.pipeline.run_pipeline` trains, scores, neutralizes and writes artifacts
but contains no upload; `zemir.pipeline.submit_predictions` is the only thing in
the package that uploads, and only `scripts/run_pipeline.py` calls it. An
experiment therefore cannot fire a live submission by forgetting a flag.

The `MIN_VALIDATION_MEAN_CORR` gate lives with the submission it guards, in
`scripts/run_pipeline.py` — not in the pipeline, which has nothing to stop.

`scripts/run_pipeline.py` is what `.github/workflows/weekly-pipeline.yml` runs
Tue–Sat at 14:00 UTC.

## Measuring: the harness

`run_pipeline` scores each model's *raw* validation predictions — an artifact
nobody uploads. The harness in `zemir/harness.py` scores the one that does get
uploaded: combine → neutralize, on Numerai's **paid** metrics.

It is split at the only expensive seam, so a comparison never pays to refit:

| Command | Cost | Writes |
| --- | --- | --- |
| `python scripts/fit_harness.py --model ensemble` | ~1 h | `runs/harness/<run_id>/validation_predictions.parquet` + `fit_config.json` |
| `python scripts/fit_harness.py --model ensemble --smoke` | seconds | the same, at meaningless scale |
| `python scripts/score_harness.py [run_id]` | minutes | `scores.csv`, `era_corr.csv`, `era_mmc.csv`, `era_max_feature_corr.csv`, `scoring_config.json` |

Fitting caches every model's raw validation predictions once. Scoring sweeps
combinations and neutralization proportions over that cache and can be re-run
freely. **Anything that changes a model** — hyperparameters, feature set,
training eras — invalidates the cache and needs a new fit; anything downstream
of `.predict()` does not. `score_harness.py` reads the cache's own
`fit_config.json`, so a sweep cannot be scored against the wrong dataset.

### What it measures, and why those metrics

Numerai pays `0.75 * corr20 + 2.25 * mmc20` — **MMC is weighted three times
CORR**. `score_validation`'s plain Spearman is not a payout metric, so the
harness uses `numerai-tools`, Numerai's own reference implementation, for both:

- **`mean_corr`** — `numerai_corr` per era over the full validation span.
- **`mean_mmc`** — MMC against `meta_model.parquet`, which covers a *window* of
  validation (96 eras in v5.0), not all of it. `mean_corr_window` is CORR
  restricted to those same eras, so the two halves of `payout` are comparable.
- **`payout`** — the weighted combination, and the headline number.
- **`max_feature_corr`** — largest absolute feature exposure per era, so
  neutralization's effect is visible rather than inferred.
- **`sharpe` / `smart_sharpe`** — risk read-outs. Not payout metrics.

`rank_normalize` is deliberately not applied: all of the above rank internally,
so it is provably free (verified to `0.00e+00`).

A `ScoringConfig` names one row of the comparison table — which models, and at
what neutralization proportion. `proportion = 0.0` *is* "no neutralization".

## Configuration

`zemir/config.py` holds one flat, frozen `PipelineConfig` and named profiles of
it. `LIVE` is what the scheduled workflow runs; `SMOKE` is the same code path at
a scale that finishes in seconds. A sweep builds its own variants:

```python
from dataclasses import replace
from zemir.config import LIVE
candidate = replace(LIVE, neutralization_proportion=0.25)
```

Every run writes a `config.json` next to its scores recording the config, the
models fitted, and the neutralizer count — so a comparison between two runs is
attributable rather than folklore.

## Scale: smoke vs full

A full era-boosted run refits XGBoost 40 times over 2.7M training rows and takes
roughly an hour. `SMOKE` trims training and validation to their 40 most recent
eras and the boosting loop to 2 iterations, running the identical code path in
about 12 seconds. Use it to check that a change *runs*; its scores are
meaningless and must never be compared against a full-scale run.

Approximate wall-clock, 8-core machine:

| Run | Time |
| --- | --- |
| `--model linear` (full) | 15 s |
| `--model ensemble --smoke` | 12 s |
| `--model ensemble` (full) | ~1 h |

## Running locally

```bash
python -m venv ~/ml-venv                     # once
~/ml-venv/bin/pip install -e prod/zemir_01   # once
cd prod/zemir_01
~/ml-venv/bin/python scripts/run_experiment.py --model ensemble --smoke
```

## Data

Datasets are cached in the repo at `datasets/<version>/` (gitignored, ~6 GB for
v5.0). `train.parquet` and `validation.parquet` are downloaded once and reused;
`live.parquet` is re-downloaded every run, because a new round replaces the
previous round's features under the same filename. `meta_model.parquet` (15 MB)
is the crowd prediction MMC is measured against and is only read by the harness.

Credentials live in `prod/zemir_01/.env` (see `.env.example`) and are read only
by `submit_predictions` — an experiment never touches them.
