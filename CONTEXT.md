# Numerai Classic Pipeline

`prod/zemir_01`'s domain: downloading Numerai Classic tournament data, training models against a scoring target, and submitting predictions for payout.

## Language

**Payout target**:
The specific target column (e.g. `target_ender_20`) that Numerai actually scores submissions against for payouts and the leaderboard, as announced by Numerai independently of any dataset release. Changes over time on Numerai's own schedule (most recently to Ender20, effective the round starting 2026-01-01).
_Avoid_: assuming this is whatever the dataset's `target` alias currently points to — confirm against Numerai's own announcements.

**`target` alias**:
A generic, backward-compatible column every dataset version provides under the literal name `target`, distinct from the many explicitly-named target columns (`target_ender_20`, `target_ender_60`, etc.) in the same file. Which named target it aliases changes across dataset versions and is **not guaranteed to match the current payout target** — verified in v5.3 to alias `target_ender_60` while Numerai pays on `target_ender_20`, a divergence that went unnoticed for months (see [issue #64](https://github.com/vltnkiz/numerai/issues/64)).
_Avoid_: treating `target` as self-evidently correct; check what it aliases against the payout target before trusting it.

**Feature exposure** (`max_feature_corr`):
The largest absolute per-era correlation between the submitted prediction and any single feature. A **diagnostic, not a priced quantity**: verified against Numerai's own docs that it enters *nothing* on Numerai Classic — not the payout formula (`0.75·corr20 + 2.25·mmc20`), not burn, not the payout factor (a pure `stake_threshold / total_at_risk` ratio), not stake eligibility or the stake cap (a flat NMR limit), and not meta-model weighting (purely stake-proportional). FNC, which measures exposure-adjusted skill, is explicitly "an informational score not used for payouts", and TC — the one metric that ever transmitted the portfolio optimizer's risk-factor penalties back to participants — is gone from the payout formula and no longer listed among the scores at all (see [issue #66](https://github.com/vltnkiz/numerai/issues/66)).
_Avoid_: trading measured payout away for lower exposure, or treating a high-exposure configuration as carrying an unbooked cost. Numerai's claim about exposure is that it *predicts* out-of-sample decay, not that it is charged for — and realized out-of-sample decay is already what the harness's 655-era CORR/MMC measurement captures directly. Report exposure; decide on payout. (Exposure *is* priced on Numerai **Signals**, via its `Alpha` metric — a different tournament, different factors, and not applicable here.)

**Numerai CORR**:
The correlation Numerai pays on, measured per era: prediction and payout target are each ranked and gaussianized, the prediction is raised to the power 1.5, and the two are correlated. One of the two priced halves of payout (`0.75·corr20 + 2.25·mmc20`); MMC is the other, and carries three times the weight. Taken from `numerai-tools`, Numerai's own reference implementation, rather than reimplemented.
_Avoid_: writing or reading a bare `mean_corr` without saying which vocabulary produced it — **Spearman correlation** reports a column of that exact name holding a different number.

**Spearman correlation**:
Plain per-era rank correlation between a prediction and the target. Numerai prices nothing on it. It survives in this project for one reason: the **history log** recorded `mean_corr`, `sharpe` and `smart_sharpe` in this vocabulary from [issue #68](https://github.com/vltnkiz/numerai/issues/68) until the live run moved to the **paid metrics**, and the live run keeps recording it on the **raw blend** until the last of those entries ages out (see `docs/adr/0002-the-live-run-gates-on-one-number-and-records-the-rest.md`). On the same production run it reads 0.0164 on the raw blend where **Numerai CORR** reads 0.0116 on the **submitted blend**, so the gap is a difference of metric **and** of artifact (see [issue #96](https://github.com/vltnkiz/numerai/issues/96)).
_Avoid_: "validation score" (the retired name, from the days it gated the submission), comparing a Spearman number against a Numerai CORR number, and measuring anything new on Spearman at all.

**Paid metrics**:
The three numbers Numerai pays on, and the vocabulary the harness ranks strategies in: **Numerai CORR**; MMC, what a prediction adds to the target's correlation beyond what the meta model's own prediction already accounts for; and payout, `0.75·CORR + 2.25·MMC`, with both halves taken over the **meta-model window**. The gate reads only the first. Payout is recorded and never gates (see `docs/adr/0002-the-live-run-gates-on-one-number-and-records-the-rest.md`).
_Avoid_: "score" or "validation score" for any one of them. Say which metric.

**Meta-model window**:
The validation eras for which Numerai has published its meta model's predictions: the only eras over which MMC, and therefore payout, can be measured. Shorter than validation, and it grows by about one era a week as the shared copy is refreshed, but it never rewrites an era already in it (see [issue #94](https://github.com/vltnkiz/numerai/issues/94) and [issue #101](https://github.com/vltnkiz/numerai/issues/101)). Numerai CORR over all of validation does not depend on it.
_Avoid_: comparing MMC or payout between two measurements taken over different windows. A longer window is not a better score, only a different one; check `mmc_eras` first.

**Strategy**:
A named list of `ModelSpec` plus one `BlendSpec` — which models, blended how. Named `zemir_NN` when it is a shipped or ship-candidate blend; named after its single model otherwise (`linear`, `era_boost`). A generation label, **not** a submission slot: a strategy can be fitted and measured before, or without, ever shipping (see [issue #74](https://github.com/vltnkiz/numerai/issues/74)).
_Avoid_: assuming a strategy's name implies it is what production currently submits — check `PRODUCTION_STRATEGY` for that.

**Neutralization**:
Removing a chosen proportion of a prediction's linear component explained by a chosen set of features, era by era. A strategy can apply one to each model's predictions before they are blended, and one to the blend afterwards; each carries its own strength and names its own features, and a stage with none applies nothing. There is no default feature set: what a neutralization removes exposure to is always written down where it is asked for, so the live run and the harness cannot disagree about what "all features" means (see [issue #88](https://github.com/vltnkiz/numerai/issues/88)).
_Avoid_: reading "no neutralization" and "neutralize against everything" as the same absence — the first is a stage that applies nothing, the second is a set that must be named.

**Raw blend** (`combined`):
The strategy's models' predictions rank-blended by their weights, before any **neutralization** at all. Never submitted. Scored only so each run can be compared with the history log's older entries, which were all taken on it (see [issue #97](https://github.com/vltnkiz/numerai/issues/97)).
_Avoid_: reading a `combined` score as a measurement of what was submitted.

**Submitted blend** (`combined_neutralized`):
The blend after every neutralization the strategy applies (each model's, then the blend's), which is what the live run uploads. The submission gate reads its **Numerai CORR** over validation. It is also the one row of a live run that measures the same thing as a harness row for the same strategy, so it is the row to compare with a harness table (see [issue #97](https://github.com/vltnkiz/numerai/issues/97)).
_Avoid_: gating on, or comparing against the harness with, the **raw blend**. The two differ by every neutralization the strategy applies.

**History log** (`score_log.jsonl`):
One entry per live run, kept for a year, recording what that run scored. Entries from before the live run moved to the **paid metrics** are **Spearman correlation** on the **raw blend** and nothing else. Later entries carry the paid metrics of each model, the raw blend and the submitted blend, and keep the Spearman figures beside them only so the old series continues (see [issue #98](https://github.com/vltnkiz/numerai/issues/98)).
_Avoid_: comparing anything in the history log with a harness table except the **submitted blend**'s paid metrics. That includes every older entry's `mean_corr`: it has the same name as a harness column but it is a different metric on a different artifact.

**Fitted model**:
The result of fitting a `ModelSpec`: a trained model bound to the exact feature columns it was fitted on, so it predicts from any frame containing them and cannot be handed the wrong ones silently. The spec is the recipe; the fitted model is what it produced, and carries no memory of which spec that was (see [issue #84](https://github.com/vltnkiz/numerai/issues/84)). The harness keeps each one beside the predictions it produced, so a question about a fit is answered from the saved model rather than a refit (see [issue #87](https://github.com/vltnkiz/numerai/issues/87)).
_Avoid_: "fitted spec" (the retired name — it held a spec nothing read, and was never a spec).

**Explanation**:
What a fitted model reports about which of its own columns it leans on, in its estimator's native quantities: a coefficient for a linear model; split count, mean gain and mean cover for a booster. A description of *this fit*, exact by construction, not an estimate of what matters in the data — at production settings a booster's explanation barely reproduces across seeds and a linear one does not reproduce across time (see [issue #83](https://github.com/vltnkiz/numerai/issues/83)). Not comparable across trainers.
_Avoid_: "feature importance", and reading a ranking from one fit as evidence a feature matters.

**Feature set**:
A named list of feature columns from a dataset version's `features.json` (`small`, `medium`, `all`). `ModelSpec.features` names the one a model is fitted and predicted on; `PipelineConfig.feature_set` names the run's universe for scoring — the set the harness measures feature exposure against — and is not a neutralization default: a **neutralization** names its own features. They are deliberately separate, so a strategy can hold models at different widths (see [issue #80](https://github.com/vltnkiz/numerai/issues/80)). A run loads the **union** of every model's columns (plus any columns a neutralization names) once, and each model is handed only its own slice.
_Avoid_: reading `PipelineConfig.feature_set` as the width a model trained on — check the model's own `features` — or as what anything is neutralized against. Feature sets are not nested: `small` is not a subset of `medium` (only 11 of its 42 columns appear there), so a strategy naming both loads more than `medium`.

**Submission slot**:
A Numerai model id that `submit_predictions` uploads to, resolved through `NUMERAI_MODELS`. `SUBMISSION_MODEL_SLOT` names a slot; a **strategy** names what gets fitted. They are deliberately separate — `SUBMISSION_MODEL_SLOT = "zemir_01"` names a slot, `STRATEGIES["zemir_01"]` names a strategy — so a strategy can exist, and be measured, without occupying or implying a slot.
_Avoid_: conflating a strategy name with a submission slot name just because they currently share a spelling (`"zemir_01"`); one is a fit, the other is where predictions get uploaded.
