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
