# Deviations and additions to PREREG.md

Each entry: time (CEST), what, why, whether it touches the verdict. The PREREG.md rule is still reported.

## 2026-10-17 15:03 CEST — reporting additions, before any attack run (no verdict change)

1. Next to every R^2 (primary, sensitivity rows, baselines), report the SD of the measured drops over the
   scored pairs and the count of pairs with |measured drop| > 0.02. Reason: Neural Chameleons
   (arXiv 2512.11949) finds evasion highly selective, so most off-target drops may be near 0 (Iris #1661,
   Tove #1671). If most drops are near 0, the post says the test had little variance to explain, and a
   NULL then means "little transfer", not "wrong geometry".
2. Exploratory diagnostics, foldout only: log `u'1 / |u|` for every probe (1 = all-ones vector in R^512),
   and add the projection of the measured class-mean edit on `1/sqrt(512)` to the edit-direction
   diagnostic. Reason: every path from layer 3 to the logits passes a LayerNorm that subtracts the
   per-token mean, so the all-ones direction is an exact null of the pull-back Fisher (Iris #1674,
   research/ln_null_check.out). An attacker can move along it at zero output cost.
