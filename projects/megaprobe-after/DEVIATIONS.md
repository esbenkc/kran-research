# DEVIATIONS — megaprobe-after

Changes after the PREREG freeze (commit ad300da, 2026-10-20 15:00:34 CEST). Each header time is the git commit
time of the entry. The frozen rule is still reported where it differs.

## D1 — 15:02:20 CEST (commit f960c97) — implementation details left open by PREREG (Felix, from Bruno #3753; no probe fit and no attacked score existed)

- `mlp14`: minibatch 64 rows, shuffled with seed 1020.
- `mega`: every fit (grid folds, label-curve fits, the main 1400-row fit) uses 400-row minibatches; T is padded to
  the batch's longest statement (padding masked). The inner 20% split is by city, seed 1020.
- Label-curve subsets: the first 50 / 100 / 200 / 350 cities of one seed-1020 permutation of the 350 training-pool
  cities (200 / 400 / 800 / 1400 statements; nested; every city brings its 2 true and 2 false statements, so each
  subset is class-balanced).
- Reproduction gate: an extra, stricter check is added; the frozen one stays. Frozen (PREREG s1): eval AUROC of the
  frozen 1019 probe within 0.005 of 1019 per seed. Added: max |score difference| <= 1e-3 against the 1019 logged
  scores on all 2992 statements per state. A run passes only if both hold.
- Can it move the headline? No: these fill gaps without changing a metric, split, threshold or status rule.

## D2 — 15:17:54 CEST (commit f94d55c) — exploratory controls added AFTER the first result (Tove #3766; Felix logs). Outside the verdict.

The first result (Bruno #3763, commit 8926df8) was seen before this entry. These arms do not change the status
rule (PREREG s6) or the prediction outcome (s7). They decide only how the post explains the megaprobe's catch.

- (A) one-layer megaprobe: the `mega` model and fit recipe (PREREG s3, D1; HP from data/hp.json, k=32, L2=1e-2,
  not re-selected) with its input restricted to ONE layer, at l = 18 and at l = 22 (the clean and the attacked
  top-attention layers). Attention runs over tokens only.
- (B) all-layer linear, no token selection: mean over tokens 1..n-1 at each of l = 1..28, concatenated
  (28 x 1536), standardised, logistic regression, C by the `lin_l` rule (PREREG s4).
- Both fit on the clean training pool only, scored on every state, PREREG s5 line and CI.
- Reading fixed now, before either arm runs:
  - If (A) at l18 or l22 has pooled attacked lower CI >= 50%: the post says token selection in the upper half does
    most of the work, not the number of layers.
  - If (A) at both layers and (B) have pooled attacked catch < 30%: the post may say that only the combination of
    layers and token selection recovers the catch.
  - Otherwise: the post says both parts contribute, with the three numbers.
- Sensitivity (reported, not the verdict): `mega` catch with the line set at the 1% point of the 596 eval true
  statements of each state (k = floor(0.01 x 597) = 5th-highest), since seed 2 had 2.0% eval false alarms at the
  refit line.
