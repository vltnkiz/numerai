# Feature Neutralization Resources

## Knowledge

- [Forum: "Model Diagnostics: Feature Exposure" — Numerai Forum](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899)
  Introduces feature exposure (Spearman correlation between predictions and each feature), why high exposure predicts poor out-of-sample performance on Numerai, and the correlation/exposure tradeoff. Use for: motivating *why* exposure matters, the diagnostic formulas (max and RMS feature exposure).
- [Forum: "An Introduction to Feature Neutralization & Exposure" — Numerai Forum](https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955)
  Explains the neutralization procedure itself: decomposing predictions via least-squares regression against feature exposures (Moore-Penrose pseudoinverse), subtracting a `proportion`-scaled component, rescaling. Use for: the mechanics/math of neutralization and the `proportion` parameter.

- [numerai-tools `scoring.py` — `neutralize` function (GitHub)](https://github.com/numerai/numerai-tools/blob/master/numerai_tools/scoring.py)
  The official, current implementation of neutralization: `np.linalg.lstsq` against a feature matrix with an appended intercept column, scaled by `proportion` and subtracted. Use for: grounding the code-level lesson on implementing neutralization against a real prediction/feature matrix.
- [3Blue1Brown — "Matrix multiplication as composition" (Essence of Linear Algebra, ch. 4)](https://www.youtube.com/watch?v=XkY2DOUCWMU)
  Geometric intuition for matrix-vector and matrix-matrix multiplication. Use for: the `Xβ` step inside the neutralize function.
- [3Blue1Brown — "Inverse matrices, column space and null space" (Essence of Linear Algebra, ch. 7)](https://www.youtube.com/watch?v=uQhTuRlWMxw)
  Geometric intuition for what an inverse matrix is and when one exists (square matrices only). Use for: motivating why non-square feature matrices need a pseudo-inverse instead.

## Wisdom (Communities)

- [Numerai Forum](https://forum.numer.ai)
  The primary community for Numerai model-building discussion; both knowledge sources above are threads from it. Use for: troubleshooting model-specific exposure/neutralization results, seeing how other participants tune the `proportion` parameter.

- [numerai/example-scripts — `feature_neutralization.ipynb`](https://github.com/numerai/example-scripts/blob/master/numerai/feature_neutralization.ipynb)
  The official walkthrough notebook (local copy in `example-scripts/numerai/`). Shows both `numerai_corr` and `neutralize` called inside `groupby("era").apply(...)`. Use for: grounding why CORR and neutralization are computed per-era rather than pooled across the dataset.

## Gaps
- Filled: the `numerai-tools` `neutralize` function and era-wise looping (both CORR and `neutralize` inside `groupby("era")`) are now covered. Still open: applying this end-to-end to the user's own trained model/predictions, and the tournament's actual scoring formula details (MMC, payout curve) — explicitly out of scope per MISSION.md unless it becomes necessary.
