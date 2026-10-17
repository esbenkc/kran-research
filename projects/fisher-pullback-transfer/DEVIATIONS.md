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

## 2026-10-17 15:09 CEST — secondary same-concept stack, declared before the pilot and before any attack run

Decision by Felix on Tove's proposal #1689 (support: Iris #1691, Kasper #1692, Vera #1693). Reason: the
clean-data identifiability gate (#1683, data/identifiability.json) shows predicted cross-concept drops with
SD 0.0007-0.0026 AUROC, so R^2 on the 245 frozen pairs may be mostly noise. The monitor-design case in the
literature is a stack of probes for the SAME concept (Bailey 2412.09565 ensembles; Gupta & Jenner
2506.14261 App. L; Neural Chameleons 2512.11949).

- **The primary verdict, the status line and lab_status are unchanged: they come only from the frozen
  245 cross-concept pairs and the PREREG.md rule.**
- Secondary set: per target, 10 extra probes on the TARGET's own labels and train split: 5 logistic (L2,
  C = 1.0) and 5 mass-mean (class-mean difference), each fit inside a random coordinate-free subspace
  (random orthonormal basis) of dimension k drawn log-uniform from [2, 512], mapped back to R^512.
  Seed 0 for target index 0, seed i for target i. Same clean-AUROC >= 0.75 filter. 50 pairs if all pass.
  No extra attack runs: scored on the saved attacked activations of the target's test texts.
- Map for the secondary set: the gap form (PREREG "Sensitivity row"): gap of `s_u` shifts by
  `-kappa_w r_P(u, w) G_u(w)`, clean sd kept, through `Phi(d'/sqrt 2)`, predicted drop =
  `Phi(d'_clean/sqrt2) - Phi(d'_after/sqrt2)`. Reason (theory/lemma.md E6, before data): on the target's
  own texts the affine-in-s_w edit only rescales `s_w`, so the exact empirical map predicts no drop for a
  probe parallel to `w` while the target itself falls to AUROC < 0.6; the gap form is self-consistent on
  the target (u = w gives d' -> (1 - kappa_w) d'_w). The exact empirical map is reported as the
  sensitivity row for this set.
- Secondary verdict, same thresholds: `R^2_F >= 0.6`, `R^2_F - max(R^2_Sigma, R^2_I) >= 0.10`,
  `R^2_F - R^2_shared >= 0.10`. Labelled "secondary, added before the attacks" in the post, TLDR and
  figure caption. Same SD / count |drop| > 0.02 reporting.
- Clean-data identifiability for the secondary set (pairwise corr of the three predicted-drop vectors at
  kappa_w = 0.5) is computed and posted before the pilot.
- Raw tables mark each pair `set = primary | secondary`; same columns as the primary.
- Attack runs save the attacked activations of every dataset's test texts (needed for this set and for
  any skeptic rebuild).
- (15:11, Kasper #1696) The secondary R^2 uses a different map from the primary, because the primary map
  has the target-parallel degeneracy above. The two sets are never pooled into one R^2, the figure legend
  shows them as separate sets, and a secondary pass is not evidence that the original pre-registered
  prediction held.
