# Research: live/current-round data handling and direct submission via numerapi

Ticket: [#4 Live/current-round data handling](https://github.com/vltnkiz/numerai/issues/4), part of map [#1 Zemir Numerai Pipeline](https://github.com/vltnkiz/numerai/issues/1).

Scope: research only. No pipeline code changed. Sources consulted:

- Installed package: `numerapi==3.1.1` at `~/ml-venv/lib/python3.11/site-packages/numerapi/` (`numerapi.py`, `base_api.py`, `utils.py`, `cli.py`, `signalsapi.py`).
- Repo notebook: `example-scripts/numerai/hello_numerai.ipynb`.
- Repo script: `scripts/download_data.py`.
- Ground-truth data actually present in this repo's working tree: `example-scripts/numerai/v5.3/live.parquet` and `.../v5.3/features.json`, `datasets/5.0/train.parquet`, `datasets/5.0/validation.parquet` — read directly with pandas/pyarrow rather than assumed.

---

## 1. Live features exposure

**How `NumerAPI` exposes current-round live data.** There is no dedicated "get live data" method distinct from the general dataset mechanism. Live data is just another named file fetched through the same generic pair of calls used for every other dataset:

- `Api.list_datasets(round_num=None)` — runs a `listDatasets(round, tournament)` GraphQL query and returns the flat list of available filenames for a round, defaulting to the current round when `round_num` is omitted (`numerapi/base_api.py:154-178`, docstring at :155-169).
- `Api.download_dataset(filename, dest_path=None, round_num=None, filters=None)` — resolves a signed download URL via a `dataset(filename, round, tournament)` GraphQL query, then streams it to disk (or, if `filters` is given, reads it through `fsspec`/`pd.read_parquet(filters=...)` and re-writes only the matching rows) (`numerapi/base_api.py:180-257`). Nothing in this method special-cases "live" — the caller supplies the live filename like any other.

**Filename convention.** The installed package itself does not hardcode a "live" filename for Numerai Classic (the string `"live"` never appears in `numerapi.py`/`base_api.py`). Two things in the installed package do reference "live" naming, both informative but not authoritative for Classic v5:
- `numerapi/signalsapi.py:240` — the **Signals** API subclass calls `self.download_dataset("signals/v1.0/live.parquet")`, i.e. Numerai's own client code uses a versioned `<version>/live.parquet` path pattern for that adjacent tournament.
- `numerapi/cli.py:99,110` — the CLI's `download_dataset` command defaults `--filename` to `"numerai_live_data.parquet"` (an older, unversioned filename convention, likely a holdover from a pre-v4 datasets layout, not what v5 clients use).

The actual, currently-used convention for Classic is confirmed independently by two primary sources in this repo:
- `scripts/download_data.py:23-33,66-70,123` — `AVAILABLE_DATASETS` includes `"live.parquet"`, `"live_benchmark_models.parquet"`, `"live_example_preds.parquet"`/`.csv`, and `download_datasets()` builds the remote path as `f"v{version}/{filename}"` (`scripts/download_data.py:123`), i.e. `v5.0/live.parquet` for version 5.0.
- `example-scripts/numerai/hello_numerai.ipynb`, cell 35 — `napi.download_dataset(f"{DATA_VERSION}/live.parquet")` with `DATA_VERSION = "v5.3"` (set in cell 6), i.e. `v5.3/live.parquet`.

So: **`download_dataset` is filename-agnostic; the live file's name/path is a convention (`v{version}/live.parquet`) observed in this repo's own script and the official example notebook, not something the numerapi library enforces or documents as a constant.**

**Shape/columns vs. `train.parquet`/`validation.parquet`.** Verified directly by reading the parquet files present in the repo, rather than inferred:

| | `datasets/5.0/train.parquet` | `datasets/5.0/validation.parquet` | `example-scripts/numerai/v5.3/live.parquet` |
|---|---|---|---|
| column count | 2416 | 2416 | 3599 (different version, v5.3 has more features/targets than v5.0 — not a live-vs-train structural difference) |
| has `era` | yes | yes | yes — but every row's `era` value is the literal string `"X"` (checked: `live["era"].unique() == ["X"]`) |
| has `data_type` | yes | yes | yes — value is `"live"` for every row (`live["data_type"].unique() == ["live"]`, vs. `"validation"` for the validation file, per notebook cell 24's filter on `data_type == "validation"`) |
| has `target` and all versioned target columns (`target_agnes_20`, `target_alpha_60`, ... 41 targets in v5.3) | yes | yes | **yes — same target columns exist in the schema**, but every value is null (`live["target"].isna().all() == True`, dtype `float32`) |
| feature columns | same `feature_*` set as `features.json` for that version | same | same `feature_*` set as `features.json` for that version (3555 feature columns in v5.3, matching `feature_metadata["feature_sets"]["all"]` length exactly) |
| index | `id` | `id` | `id` |

Key finding, confirmed from actual data (not assumed): **target columns are present in the live dataframe's schema, but are all-null placeholders** — the row shape matches train/validation exactly (same feature columns keyed against that version's `features.json`), only `era` is fixed at `"X"` and `data_type` is `"live"` to mark it as the unresolved current round. This matters for pipeline code: a naive `dropna(subset=["target"])` or a schema check for target *values* will behave very differently on live data than on train/validation, but a check merely for target *column presence* would not distinguish live from train.

The example notebook's own live-inference cell only reads back the feature columns anyway: `pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=feature_set)` (`hello_numerai.ipynb`, cell 35) — it doesn't touch `target` at all when generating predictions, sidestepping the null-target issue by construction.

---

## 2. Round timing

**Discoverable, with caveats, from the installed package + the notebook's prose, not purely inferred:**

- `Api.get_current_round(tournament=None)` — queries `rounds(tournament, number: 0)`("zero is an alias for the current round", comment at `numerapi/base_api.py:718`) and returns just the round `number` (`numerapi/base_api.py:703-732`). It does **not** return timing fields.
- `Api.list_rounds(number=None, target=None, status=None, limit=None)` — the richer query. Requesting rounds (e.g. with `status="open"` or a specific `number`) returns, per row: `closeTime`, `closeStakingTime`, `openTime`, `scoreTime`, `resolveTime`, `resolvedGeneral`, `resolvedStaking`, `payoutFactor`, `stakeThreshold`, `dataDatestamp`, plus nested `roundScoreConfigs` (`numerapi/base_api.py:734-831`). These are the concrete timestamp fields the API surface exposes for scheduling — `openTime` (when the round/live data opens) and `closeStakingTime`/`closeTime` (submission deadline window). All are parsed to `datetime` via `utils.parse_datetime_string` before being returned (`numerapi/base_api.py:814-823`).
- `Api.check_round_open()` — a convenience boolean built exactly from `openTime` and `closeStakingTime`: `is_open = open_time < now < deadline` (`numerapi/base_api.py:1704-1738`). This confirms `closeStakingTime` is the operative submission deadline, not `closeTime` or `resolveTime`.
- `Api.check_new_round(hours=12)` — checks whether `openTime` for the current round falls within the last `hours` hours (default 12) relative to `datetime.now(utc)` (`numerapi/base_api.py:1740-1775`). This is a ready-made polling primitive for a scheduled job: call `check_new_round()` and only proceed with download/submit if it returns `True`.
- `Api.get_competitions(tournament=8)` similarly returns `openTime`/`resolveTime` per round from a `rounds` query, but only `resolveTime` (not `closeStakingTime`), so it's less precise for the submission deadline than `list_rounds` (`numerapi/numerapi.py:26-75`).

None of these source files state an absolute wall-clock schedule (e.g. "Tuesdays at 18:00 UTC") as a constant — the actual cadence has to be read off live `openTime`/`closeStakingTime` values returned by a real API call, since rounds don't necessarily open at a fixed time hardcoded in the client.

**Where a concrete cadence *is* stated, in prose, in a primary source in this repo:** `example-scripts/numerai/hello_numerai.ipynb`, cell 34 (markdown): *"Every Tuesday-Saturday, new `live features` are released, which represent the current state of the stock market."* And cell 36 (markdown): *"To participate in the tournament, you must submit live predictions every Tuesday-Saturday."* This is Numerai's own official example notebook, so it's a credible primary source for the cadence, but it is prose in a tutorial, not a machine-readable constant in the API client — no `numerapi` source file encodes "Tuesday-Saturday" or specific UTC hours anywhere.

**Gap — flagged for manual confirmation:** The exact day-of-week/time-of-day boundaries (which weekday the round opens on, the exact UTC time live data becomes downloadable, and the exact UTC time `closeStakingTime`/submission deadline falls) are **not hardcoded or discoverable as constants from the installed numerapi package alone**. They can only be obtained by (a) trusting the notebook's "Tuesday-Saturday" prose, or (b) calling `list_rounds(status="open")` (or similar) against the live API and reading the actual `openTime`/`closeStakingTime` timestamps for the current round, or (c) checking Numerai's website/docs directly. For scheduling ticket #10's weekly GitHub Actions job, the safe approach given only what's confirmed here is: schedule the cron generously inside the Tuesday-Saturday window and then gate the job's actual download/submit logic on `check_new_round()` / `check_round_open()` at runtime, rather than hardcoding a single trigger time — because this research did not find a documented, stable, single "data drops at this exact time" guarantee in the primary sources available.

---

## 3. Submission call format

`Api.upload_predictions` — the direct-upload method that replaces the cloudpickle/compute-cluster approach the notebook currently demonstrates (`numerapi/base_api.py:2129-2207`):

```python
def upload_predictions(
    self,
    file_path: str = "predictions.csv",
    model_id: str | None = None,
    df: pd.DataFrame | None = None,
    data_datestamp: int | None = None,
    timeout: Union[None, float, Tuple[float, float]] = (10, 600),
) -> str:
```

Docstring (`numerapi/base_api.py:2137-2161`):
> "Upload predictions from file. Will read TRIGGER_ID from the environment if this model is enabled with a Numerai Compute cluster setup by Numerai CLI."
> - `file_path (str)`: CSV file with predictions that will get uploaded
> - `model_id (str)`: Target model UUID (required for accounts with multiple models)
> - `df (pandas.DataFrame)`: pandas DataFrame to upload, if function is given df and file_path, df will be uploaded.
> - `data_datestamp (int)`: Data lag, in case submission is done using data from the previous day(s).
> - Returns: `str`: submission_id

Behavior read from the source body:

- **Input format**: it accepts *either* a CSV file path *or* a DataFrame passed via the `df=` kwarg — not both simultaneously in effect, since `df is not None` takes precedence: `if df is not None: buffer_csv = BytesIO(df.to_csv(index=False).encode()); buffer_csv.name = file_path` (`base_api.py:2166-2170`). So a DataFrame is accepted directly (no need to write a temp CSV to disk first), but internally it is still serialized to CSV bytes via `df.to_csv(index=False)` before uploading — **`index=False`**, meaning if the DataFrame's index carries the `id` column (as `live.parquet` does — index name `id`, confirmed above), the caller must reset/include it as an actual column before calling, or the `id`s will be silently dropped from the upload. This is a concrete pitfall for ticket #4/#10's pipeline code.
- **`model_id`**: passed straight through into `_upload_auth("submission_upload_auth", file_path, self.tournament_id, model_id)` (`base_api.py:2172-2174`) as a GraphQL argument (`base_api.py:989-996`, `modelId` in the mutation `arguments` at `base_api.py:2201`) — required for accounts with multiple models per the docstring, but the signature itself allows `None`.
- **Round handling**: there is no explicit `round_num` parameter. The round is implicit — the server presumably associates the upload with whatever round is currently open/accepting submissions at request time (consistent with `check_round_open()`'s semantics above); numerapi's client code does not pass a round number into the `create_submission` mutation (`base_api.py:2182-2205`) — only `filename`, `tournament`, `modelId`, `triggerId` (from `os.getenv("TRIGGER_ID")`, `base_api.py:2202`, only set in Numerai-Compute-managed environments), and `dataDatestamp`.
- **Upload flow**: two network calls — (1) `_upload_auth(...)` fetches a signed `{filename, url}` pair via GraphQL (`base_api.py:974-996`); (2) a raw `requests.put(upload_auth["url"], data=file.read(), headers=headers, timeout=timeout)` PUTs the CSV bytes to that signed URL, with a compute-cluster header `x_compute_id` set from `NUMERAI_COMPUTE_ID` env var if present (`base_api.py:2176-2181`); then (3) a `create_submission` GraphQL mutation registers the uploaded file (`base_api.py:2182-2206`), and its `data.create_submission.id` is returned as the `submission_id` string (`base_api.py:2206-2207`).
- **Success/failure signaling**: on success, returns the `submission_id` string (e.g. `'93c46857-fed9-4594-981e-82db2b358daf'` per the docstring example). On failure, the method itself has **no try/except** — errors surface as exceptions from underlying calls: `raw_query` raises `ValueError(err)` if the GraphQL response contains an `"errors"` key (`base_api.py:148-151`, message extracted via `_handle_call_error` at `base_api.py:72-83`), or `requests.put` can raise its own `requests.exceptions.*` (timeouts, connection errors) since its response is not checked (`base_api.py:2179-2181` — note the PUT response isn't inspected for a non-2xx status at all, so an unrecognized network failure at that HTTP layer would not necessarily raise). Also: `raw_query(..., authorization=True)` raises `ValueError("API keys required for this action.")` if the client wasn't constructed with `secret_key`/`public_id` (`base_api.py:132-137`), which is what `_upload_auth` and the `create_submission` mutation both require (`base_api.py:994`, `base_api.py:2205`).
- **Note on `hello_numerai.ipynb`**: the notebook does **not** demonstrate `upload_predictions` at all — its "Submissions" section (cells 34-38) only shows the cloudpickle-based model-upload path (serializing a `predict` function with `cloudpickle.dumps` for Numerai's hosted compute), which is exactly the approach ticket #4 wants to move away from. So there is no in-repo example of the direct-upload call in practice; the signature/behavior above comes entirely from reading `base_api.py` directly.

---

## Summary of gaps requiring manual confirmation

1. **Exact round-open/close wall-clock schedule** (weekday + UTC time) is not a hardcoded constant in numerapi; only inferable from the notebook's "Tuesday-Saturday" prose or by querying `list_rounds`/`check_round_open`/`check_new_round` live. Confirm against Numerai's website/docs before hardcoding a cron schedule for ticket #10.
2. **Upload PUT response is not checked for HTTP status** in `upload_predictions` (`base_api.py:2179-2181`) — worth confirming empirically (or via Numerai support) what failure modes are actually surfaced vs. silently swallowed, since this research is based on static reading of the source, not a live failed-upload test.
3. **`round_num` targeting for `upload_predictions`** is implicit (no explicit parameter); confirm with Numerai (or by testing against a non-open round) that submissions are always scored against "whatever round is currently open" and there is no way to target a specific past/future round number directly through this method.
