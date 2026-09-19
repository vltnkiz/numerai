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

**Strategy**:
A named list of `ModelSpec` plus one `BlendSpec` — which models, blended how. Named `zemir_NN` when it is a shipped or ship-candidate blend; named after its single model otherwise (`linear`, `era_boost`). A generation label, **not** a submission slot: a strategy can be fitted and measured before, or without, ever shipping (see [issue #74](https://github.com/vltnkiz/numerai/issues/74)).
_Avoid_: assuming a strategy's name implies it is what production currently submits — check `PRODUCTION_STRATEGY` for that.

**Feature set**:
A named list of feature columns from a dataset version's `features.json` (`small`, `medium`, `all`). `ModelSpec.features` names the one a model is fitted and predicted on; `PipelineConfig.feature_set` names the run's default universe for scoring and neutralization. They are deliberately separate, so a strategy can hold models at different widths (see [issue #80](https://github.com/vltnkiz/numerai/issues/80)). A run loads the **union** of every model's columns (plus any blend neutralizers) once, and each model is handed only its own slice.
_Avoid_: reading `PipelineConfig.feature_set` as the width a model trained on — check the model's own `features`. Feature sets are not nested: `small` is not a subset of `medium` (only 11 of its 42 columns appear there), so a strategy naming both loads more than `medium`.

**Submission slot**:
A Numerai model id that `submit_predictions` uploads to, resolved through `NUMERAI_MODELS`. `SUBMISSION_MODEL_SLOT` names a slot; a **strategy** names what gets fitted. They are deliberately separate — `SUBMISSION_MODEL_SLOT = "zemir_01"` names a slot, `STRATEGIES["zemir_01"]` names a strategy — so a strategy can exist, and be measured, without occupying or implying a slot.
_Avoid_: conflating a strategy name with a submission slot name just because they currently share a spelling (`"zemir_01"`); one is a fit, the other is where predictions get uploaded.
