# Performance Stationarity & Era Boosting Resources

## Knowledge

- [Forum: "Performance Stationarity" — richai (Numerai Forum, Apr 2020)](https://forum.numer.ai/t/performance-stationarity/151)
  Introduces stationarity as a model-quality goal distinct from mean return: a good model's win/loss sequence should look "memoryless" (like a biased coin flip), not clustered into long streaks. Argues stock-market features are inherently non-stationary, which is why naive models inherit unstable era-to-era performance. Introduces "Smart Sharpe" (Sharpe with an autocorrelation penalty). Use for: motivating *why* era-to-era consistency matters, not just mean correlation.
- [Forum: "Era Boosted Models" — richai (Numerai Forum, Apr 2020)](https://forum.numer.ai/t/era-boosted-models/189)
  The primary source for the era boosting algorithm: build trees on all eras, score per-era Spearman correlation, retrain the next batch of trees only on the worst-performing half of eras, repeat. Reports in-sample Sharpe improving from 2.28 to 21.99 over 200 trees. Explicit that out-of-sample validation was not yet done at the time of posting — a caveat worth carrying into lessons. Use for: the algorithm itself, its parameters (`proportion`, `trees_per_step`/`num_iters`), and its stated limitations. Also contains the reference XGBoost implementation (warm-started via `xgb_model=booster`) used in lesson 3 and `reference/era-boosting-code.html`.

## Wisdom (Communities)

- [Numerai Forum](https://forum.numer.ai/)
  The primary community for this topic — both source posts live here, and it's where practitioners discuss whether techniques like era boosting hold up out-of-sample. Use for: testing era-boosting variants against other users' experience, checking if newer posts have superseded richai's 2020 results.

## Gaps

- No independent (non-Numerai) source yet verifying whether era boosting's in-sample Sharpe gains hold out-of-sample — the original post is explicit this wasn't tested. Worth searching the forum for follow-up threads before the user relies on this technique for a live model.
- No source yet on "Smart Sharpe" beyond the one-line formula in the stationarity post — could look for a fuller explanation of the autocorrelation penalty if a lesson needs to compute it, not just reference it.
