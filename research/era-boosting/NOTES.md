# Notes

- Sibling workspace to `../feature-neutralization/` — same user, same overall Numerai mission, same starting
  ML/stats level (beginner). Reused their `assets/lesson.css` and glossary/quiz pattern for visual continuity.
- Lesson 2 (era boosting algorithm: build → score per-era corr → retrain on worst half → repeat) is done, incl.
  a worked 6-era trace, the reported in-sample Sharpe jump (2.28 → 21.99), and the out-of-sample caveat.
- Lesson 3 (implementation: annotated XGBoost code from richai's post, incl. the xgb_model=booster warm-start
  trick, before/after Sharpe scoring, and a "your turn" run-it-yourself exercise) is done. Code reference doc
  at reference/era-boosting-code.html.
- User was asked implementation vs. out-of-sample source hunt for lesson 3; chose implementation. Follow up next
  session on whether they actually ran the "your turn" exercise and what Sharpe numbers they got before picking
  lesson 4.
- Lesson 4 candidates: (a) the out-of-sample source hunt (search Numerai forum for follow-up threads testing
  whether era boosting's in-sample gains hold out-of-sample — flagged as a RESOURCES.md gap), or (b) troubleshooting
  their actual implementation run if they hit issues.
