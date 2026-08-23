# Mission: Performance Stationarity & Era Boosting for Numerai

## Why
The user is building a model for Numerai, a crowdsourced hedge fund tournament where predictions are scored on out-of-sample stock-market data, era by era. A model with a great average score can still be dangerous to hold if its wins and losses come in long clumps — deep, multi-era drawdowns are what actually burn a stake and erode trust in a model. Understanding performance stationarity (why consistency across eras matters, not just mean correlation) and era boosting (a training technique that explicitly targets that consistency) lets the user diagnose and build models that are safer to run live.

## Success looks like
- Can explain, in their own words, why two models with the same Sharpe ratio can carry very different risk if one has clustered underperformance and the other doesn't ("memoryless" vs. autocorrelated performance).
- Can describe why stock-market features are non-stationary and why that non-stationarity leaks into a naively-trained model's era-to-era performance.
- Can explain the era boosting algorithm end to end: build trees on all eras, score per-era correlation, identify the worst-performing eras, build the next batch of trees only on those, repeat — and why this pushes the model toward equal performance across eras.
- Can implement a basic era-boosted model (e.g. with XGBoost) and compute in-sample Sharpe before/after to see the effect.
- Can look at a model's per-era correlation chart and reason about whether it looks "stationary enough" to trust.

## Constraints
- New to both ML and stats — same starting point as the sibling `feature-neutralization` workspace. Lessons should build core stats intuition (autocorrelation, Sharpe ratio, standard deviation across a sample) alongside the finance-specific content, not assume it.
- Learning is grounded in two Numerai forum posts the user identified as trusted sources (see RESOURCES.md). Prefer these over parametric knowledge.
- This is a sibling workspace to `../feature-neutralization/`, part of the same overall Numerai model-building effort — reuse compatible visual style, but treat as its own scoped mission (era boosting is a training-time technique, neutralization is a post-processing technique; don't conflate them).

## Out of scope
- General Numerai tournament mechanics unrelated to consistency (staking, scoring formulas like MMC/FNC) — revisit only if needed to explain stationarity.
- Deep dives into gradient boosting internals (how XGBoost builds a single tree) — treat XGBoost as a tool being steered, not a thing to derive from scratch, unless the user asks.
- Feature neutralization mechanics — already covered in the sibling workspace; only reference it, don't re-teach it here.
