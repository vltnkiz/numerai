# Numerai Classic Pipeline

`prod/zemir_01`'s domain: downloading Numerai Classic tournament data, training models against a scoring target, and submitting predictions for payout.

## Language

**Payout target**:
The specific target column (e.g. `target_ender_20`) that Numerai actually scores submissions against for payouts and the leaderboard, as announced by Numerai independently of any dataset release. Changes over time on Numerai's own schedule (most recently to Ender20, effective the round starting 2026-01-01).
_Avoid_: assuming this is whatever the dataset's `target` alias currently points to — confirm against Numerai's own announcements.

**`target` alias**:
A generic, backward-compatible column every dataset version provides under the literal name `target`, distinct from the many explicitly-named target columns (`target_ender_20`, `target_ender_60`, etc.) in the same file. Which named target it aliases changes across dataset versions and is **not guaranteed to match the current payout target** — verified in v5.3 to alias `target_ender_60` while Numerai pays on `target_ender_20`, a divergence that went unnoticed for months (see [issue #64](https://github.com/vltnkiz/numerai/issues/64)).
_Avoid_: treating `target` as self-evidently correct; check what it aliases against the payout target before trusting it.
