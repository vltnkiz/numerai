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

`scripts/run_pipeline.py --model ensemble` is what the Windows scheduled task
runs — see [Scheduling](#scheduling).

## Scheduling

The live run is a Windows Task Scheduler task on the 5950X desktop (32
threads, 64 GB RAM), running against this checkout. It replaced a GitHub
Actions workflow on a self-hosted WSL runner (issue #69). That runner stopped
submitting after 2026-09-05: it lost contact mid-run, then its workspace's git
repo was corrupted. Even on good days, GitHub's cron started the "12:00 UTC"
job 3–5 h late, after weekday staking had closed.

| File | Role |
| --- | --- |
| `scheduling/Register-ZemirTask.ps1` | Registers the task. Run once, interactively; it asks for your Windows password. |
| `scheduling/Invoke-ZemirLiveRun.ps1` | What the task runs: checks, pull, install, pipeline, score-log push, failure issues. `-DryRun` smoke-tests the same path without submitting. |
| `zemir/schedule.py` | Decides whether an invocation has a round to run. |

### When it runs

- **Triggers:** daily at 12:00 UTC, waking the machine and starting as soon as
  possible after a missed start, plus at logon. There's no boot trigger:
  Fast Startup turns power-on into a hibernate-resume, which never fires one.
- **Deciding whether to run:** every invocation asks Numerai for the current
  round and runs if this machine hasn't finished it. Extra triggers are
  harmless; the second one just exits 0.
- **Waiting:** only between 12:00 and 12:45 UTC on Tue–Sat does an invocation
  wait for a new round, polling every 2 min. If nothing opens, it fails loudly
  (issue #33: Numerai disclaims any bound on open-time slippage). Anywhere
  else it checks once and exits, so a logon at 08:00 doesn't sleep until noon.
- **Late rounds are still submitted:** after `closeStakingTime` Numerai scores
  an upload but won't stake it, and also queues it for the next round. This is
  also why the round check uses `rounds(number: 0)` rather than
  `NumerAPI.check_round_open()`, which is false once staking closes.
- **Round markers:** `runs/rounds/<round>.json` marks a round finished. Only
  outcomes a retry can't change write one: *submitted*, or a *validation-gate
  failure* (the fit is deterministic). A crash, network error or memory skip
  leaves no marker, and the next trigger retries.
- **Round changes mid-run:** if a new round opens during the fit, the
  predictions are not uploaded, because they were made from the previous
  round's live features.

### Guards and alerts

- **Clean `main` only:** the wrapper refuses to run from another branch or with
  uncommitted changes, so work in progress is never submitted. The cost is a
  failed run on a day the checkout isn't clean.
- **Memory:** `MIN_AVAILABLE_MEMORY_GIB` (46) is checked before the fit starts.
  The full `medium` ensemble peaked at 43.2 GiB resident on Windows. That's
  about twice issue #44's 21.48 GiB, which covered only the linear stage
  (`LinearRegression` upcasting the int8 design matrix to float64).
- **Priority and sleep:** the pipeline runs at BelowNormal priority, and the
  machine is kept awake while it runs.
- **Failure issues:** any failure opens a GitHub issue titled
  `Live run failed: <type> (round N)`, with the log tail. A repeat for the same
  type and round comments on the existing issue. Exit codes 2–5 from
  `run_pipeline.py` name the type (gate, round not open, memory, round
  changed); anything else is a crash.
- **Logs:** each invocation writes `runs/scheduled/<utc stamp>/run.log` and
  `pip_freeze.txt`. Dependencies aren't locked, so the freeze is how version
  drift is traced.
- **Score log:** `score_log.jsonl` is committed with a pathspec commit (only
  that file) and pushed, retrying once after `git pull --rebase`.

### Carried over from the workflow

- **Always `--model ensemble`:** it's passed explicitly, never left to the
  script's default (issue #32). A bare invocation once silently ran linear-only.
- **Dataset caching:** `datasets/` (~8.4 GB of v5.3) is gitignored and reused
  between runs. Only `live.parquet` is re-downloaded each round.
- **Timing:** the full `medium` ensemble takes about 22 min on this machine
  with the live download included (issue #69's parity run). The task's 150-min
  limit covers the 45-min wait plus a fit, with room to spare.

### Setup on a fresh checkout

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e "prod\zemir_01[dev]"
# Fill in prod\zemir_01\.env (see .env.example), then:
powershell -ExecutionPolicy Bypass -File prod\zemir_01\scheduling\Invoke-ZemirLiveRun.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File prod\zemir_01\scheduling\Register-ZemirTask.ps1
```

Tests for the scheduling logic: `..\..\.venv\Scripts\python.exe -m pytest`
from `prod\zemir_01`.

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
it. `LIVE` is what the scheduled task runs; `SMOKE` is the same code path at
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
