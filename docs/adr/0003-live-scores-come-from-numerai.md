# 3. Live scores come from Numerai, and nothing computed here is called payout

## Status

Accepted. Amends [ADR 0002](0002-the-live-run-gates-on-one-number-and-records-the-rest.md).

## Context

The live run printed and logged a validation `mean_corr` that was positive on
every run, and it was read as what zemir_01 was earning. Numerai's model
page showed something else. Both were right about different things. The
logged number is a backtest: a fixed model scored over the same validation
eras, so it is identical every day. Numerai scores each round live, over 20
or 60 days.

Two more facts, both read from `round_model_performances_v2` on 2026-09-26:

- **The payout formula changed.** Rounds up to 1342 list multipliers
  `0.75·v2_corr20 + 2.25·mmc` and resolve after about 30 days. Rounds from
  1343 (opened 2026-08-28) list `3.0·corr60 + 15.0·mmc60` and resolve after
  about 90. The repo hardcoded the first formula, applied to 20-day scores.
- **zemir_01 is unstaked.** Every round has `atRisk = 0`, so it pays 0 NMR
  whatever it scores.

## Decision

- **Live scores are fetched, never computed.** `zemir.live_scores` records
  Numerai's per-round scores for every slot in `NUMERAI_MODELS` in
  `prod/zemir_01/live_scores.jsonl`, with one row per (model, round). A row is
  rewritten while its round is provisional and frozen once the round
  resolves. The live run does this after the history log is appended, and it
  never changes the run's exit code. `scripts/live_scores.py` does the same
  on demand. The daily commit includes the file.
- **Payout comes from Numerai's multipliers.** Each row's `payout_score`
  uses the multipliers listed on that round. Each row's
  `reference_payout_score` uses the current formula
  (`REFERENCE_MULTIPLIERS`, `3·corr60 + 15·mmc60`) on every round, since
  Numerai reports corr60 and mmc60 on the older rounds too. The headline
  averages `reference_payout_score` over **resolved rounds only**, next to
  corr20 and mmc20. Neither score includes stake, payout factor or clipping.
- **The live run records no payout of its own.** History log schema 3 drops
  `payout` from every `paid` row, and the printed table drops it too. Every
  backtest line is labelled "validation (backtest)". The gate itself is
  unchanged.
- **The harness keeps a proxy under an honest name.** `payout` becomes
  `validation_payout_proxy`, still `0.75·corr20 + 2.25·mmc20` over the
  meta-model window. It ranks harness rows. It is not a prediction of what a
  round pays.

## Considered options

- **Compute live payout ourselves.** Rejected. Numerai owns the formula and
  has already changed it once. Reading the multipliers per round is the only
  way the number stays right without anyone noticing a change.
- **Append a daily snapshot of provisional scores.** Rejected. Provisional
  values are Numerai's to revise, not measurements of ours.
- **Average `payout_score` across formulas, or once per formula.** Rejected
  for one reference formula. The two formulas' scores differ in scale by
  about fifty times, and the current one is the only one that prices future
  rounds.
- **Move the harness proxy to the 60-day formula now.** Deferred. It needs a
  60-day target and 60-day MMC on validation, and it reopens every tuning
  decision made on the proxy.

## Consequences

- ADR 0002's "keep the two paths' column sets identical" now holds except
  for `validation_payout_proxy`, which only the harness has.
- `round_model_performances_v2` is deprecated in `numerapi` in favour of
  `submission_scores`. The latter has no multipliers and no per-round
  resolution, so this stays until Numerai removes the endpoint, and
  `numerapi` stays pinned `<4`.
- A frozen row keeps the shape it was written in. A change to the row shape
  needs a migration of the resolved rows, or a reader that tolerates old ones.
- `REFERENCE_MULTIPLIERS` is a constant. When Numerai lists new multipliers
  on new rounds, update it.
