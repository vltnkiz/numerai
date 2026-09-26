# 2. The live run gates on one number and records the rest

## Status

Accepted

## Context

Until issue #93 the live run and the harness scored in two vocabularies
that could not be compared. The live run gated on the plain Spearman
`mean_corr` of the **raw blend**, the one artifact it never submits, against
a floor of `0.001`. The harness ranks strategies on the **paid metrics**
(Numerai CORR, MMC and payout) of the **submitted blend**. Both vocabularies
wrote a column called `mean_corr`.

The review that raised this assumed payout could not be computed live
because the live round has no meta model. That was wrong. The gate scores
validation, not the live round, and the meta model covers part of validation
(issue #94). So the live run *can* compute every paid metric, including
payout, which is what Numerai actually pays. That makes this decision a
choice rather than a constraint.

## Decision

The live run computes one number before it uploads and gates on it: the
**Numerai CORR** of the **submitted blend**, averaged over every validation
era. It passes when that number is at least `MIN_VALIDATION_MEAN_CORR = 0.0`.

Everything else is computed after the upload and never gates. That covers
the full paid table (MMC, payout and exposure) for each model, the raw
blend and the submitted blend, plus the Spearman figures. If that scoring
fails, the submission stands and the run exits 6.

- **Why CORR and not payout.** MMC and payout exist only over the
  **meta-model window**, which is shorter than validation and grows about one
  era a week (issue #101). MMC is the noisier half, and payout weights it
  three times as heavily as CORR. A payout gate would be gating on the
  noisiest number, measured over the shortest span, and on a span that moves
  from one run to the next.
- **Why a floor of zero.** The gate "only catches a run crashing through the
  floor, not one quietly declining above it" (issue #68). Choosing between
  strategies is the harness's job. A floor of zero depends on no typical
  value, so it cannot go stale when the fit or the data changes. The old
  `0.001` was never shown to separate anything: the suite's only failing
  fixture was a sign flip (issue #95).
- **Why the split around the upload.** Scoring that only records can then
  never delay or break a submission, and its memory cost falls after the
  prediction peak rather than on top of it (issue #97).

## Considered options

- **Gate on payout.** Rejected for the window and noise reasons above. It is
  still computed and recorded on every run.
- **Keep the gate on the raw blend.** Rejected. It measures a prediction the
  run never uploads, and it differs from the submitted blend by every
  neutralization the strategy applies.
- **A floor derived from the production fit.** Derived first and kept as
  evidence (issue #95, 0.000630 proportion-preserving). Rejected because it
  would depend on a typical value, and that value would go stale as the fit
  and the data change.

## Consequences

- A pure-noise fit passes the gate about half the time. The gate catches a
  sign flip and nothing else. A rolling floor built from the history log's
  own series is the one mechanism left for catching gradual decline. Reopen
  this decision once the log holds enough schema 2 entries to derive one.
- The live run and the harness record the same paid columns for the same
  artifact, so a live **submitted blend** row can be compared with a harness
  row. The comparison is only exact when both were measured over the same
  meta-model window (`mmc_eras`). Keep the two paths' column sets identical:
  if `max_feature_corr` is ever dropped, drop it from both paths or from
  neither (issue #102).
- The Spearman path (`era_spearman`, `summarize_era_spearman` and the
  `spearman` blocks in the history log) is **legacy with an end condition**.
  It exists only so the raw blend's Spearman series continues through the
  schema 1 entries. Once the 365-day prune has removed the last schema 1
  entry, delete it. Until then, nothing new is measured on it (issue #96).
- **Amended 2026-09-26:** the live run now refreshes `validation.parquet`
  weekly, so the gate's frame grows past the 655 eras every schema 1 entry was
  measured on. The Spearman series continues only approximately, and nothing
  is done to preserve it: fresh data won over continuity. Where an exact
  comparison matters, recompute both sides on the current copy.

See issues #94, #95, #96, #97 and #98.
