# Mission: Feature Neutralization for Numerai

## Why
The user is building a model for Numerai, a crowdsourced hedge fund tournament where predictions are scored on out-of-sample stock-market data. Models that lean too hard on a few features perform well in backtests but "burn" badly when the market regime shifts. Understanding feature exposure and neutralization lets the user diagnose this risk and fix it — directly improving the consistency (and payout) of their submitted model.

## Success looks like
- Can explain, in their own words, why low correlation-with-target isn't enough — a model also needs low feature exposure to be robust across eras.
- Can compute feature exposure for a trained model's predictions (Spearman correlation between predictions and each feature).
- Can implement feature neutralization (regress predictions on feature exposures via pseudo-inverse, subtract a proportion of the fitted component) and explain what the `proportion` parameter trades off.
- Can look at a model's exposure/neutralization diagnostics and make a reasoned call about how aggressively to neutralize.

## Constraints
- New to both ML and stats — lessons need to build linear algebra/regression intuition (correlation, projection, pseudo-inverse) alongside the finance-specific content, not assume it.
- Learning is grounded in two Numerai forum posts the user already identified as trusted sources (see RESOURCES.md). Prefer these over parametric knowledge.

## Out of scope
- General Numerai tournament mechanics unrelated to exposure/neutralization (staking, scoring formulas like MMC/FNC) — revisit only if it becomes necessary to explain neutralization.
- Deep dives into portfolio theory / other risk models outside Numerai's specific exposure framework.
