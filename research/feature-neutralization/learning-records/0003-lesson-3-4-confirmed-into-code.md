# Lesson 3 confirmed, moving into code (Lesson 5)

The user confirmed understanding of Lesson 3's core mechanism unprompted: using a linear regression's fitted
component (found via correlation/covariance) to remove — neutralize — the part of a prediction explained by a
feature. They connected this directly to the next step themselves: applying it against a real prediction/feature
matrix, where a single feature's covariance/variance formula no longer applies and the Moore-Penrose pseudo-inverse
is needed to solve for all features' coefficients simultaneously (accounting for correlation between features).

This confirms the Lesson 1 → 2 → 3 arc (exposure motivation → Spearman measurement → single-feature neutralization
algebra) is solid. Lesson 5 was built to answer exactly the transition the user named: from a scalar β to a vector
of β's, grounded in the real `numerai-tools` `neutralize` function (see RESOURCES.md) rather than a hypothetical
implementation. Still unconfirmed: whether Lesson 4's visualization (fitted/residual scatter, perpendicularity) was
absorbed — Lesson 5 leans on that perpendicularity intuition generalizing to "perpendicular to the whole feature
space," so if that idea doesn't land, revisit Lesson 4 before going further.
