# Deviations and additions to PREREG.md

Each entry: time (CEST, = git commit time of the entry; first versions of the headers carried hand-typed times 1-7 min late, corrected 15:11), what, why, whether it touches the verdict. The PREREG.md rule is still reported.

## 2026-10-17 15:02 CEST (commit b7ab60d) — reporting additions, before any attack run (no verdict change)

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

## 2026-10-17 15:07 CEST (commit b42d20d) — secondary same-concept stack, declared before the pilot and before any attack run

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
- (15:08, commit ad0a645, Kasper #1696) The secondary R^2 uses a different map from the primary, because the primary map
  has the target-parallel degeneracy above. The two sets are never pooled into one R^2, the figure legend
  shows them as separate sets, and a secondary pass is not evidence that the original pre-registered
  prediction held.

## 2026-10-17 15:08 CEST (commit 9021dd4) — attacker loss denominator fixed to the clean sd (before the pilot and any target run)

PREREG.md "Attacker" froze loss = (batch d'_w)^2 (class-mean gap of s_w over the BATCH pooled sd). A 20-step
code smoke test (pilot probe yahoo:Entertainment & Music, lambda_KL 100; not the pilot, not a target run;
Bruno #1698) showed this loss is won by inflating the spread, not by closing the gap: test-set w-score sd
2.08/1.83 (pos/neg) -> 12.00/9.05, class-mean gap 3.17 -> 4.57 (delta_w = +1.39), AUROC 0.874 -> 0.608.
The predictor is calibrated by delta_w, so with this loss delta_w has the wrong sign and every panel would
predict that the other probes get better: a test of the loss design, not of the geometry.
New loss (Felix decision): `((mean s_w | pos) - (mean s_w | neg))^2 / sd_clean^2` on the target's train
batch, `sd_clean` = clean pooled within-class sd of s_w on the target's train set, fixed per target; plus the
same lambda_KL retain term. Everything else in "Attacker" is unchanged (pilot rule, stop rule, cap, seeds).
The smoke-test numbers are kept in the receipts.

## 2026-10-17 15:09 CEST (commit 9758dc2) — supersedes the 15:08 loss choice (still before the pilot and any target run)

The fixed-sd gap^2 loss of 15:08 optimises exactly the quantity our edit model is calibrated on, which
makes the attacker friendlier to the theory (Vera #1700, Tove #1704). Tonight's claim is about an attacker
we did not design, so the attacker follows the published obfuscation recipe instead: push the monitor's
output down on concept-positive inputs (Bailey et al. 2412.09565, Eq. 1: "L_obf = D on inputs with the
concept"; Neural Chameleons 2512.11949 Eq. 2 uses an MSE of the probe probability to 0 under the
trigger; Iris #1702).
New loss (Felix decision, Tove's procedure #1704): `mean softplus(s_w + b_w)` over target-POSITIVE train
texts (= binary cross-entropy of the frozen probe toward label 0) + `lambda_KL` * mean per-token KL on
retain windows. Gate: one 20-step smoke on the pilot probe (not a target). If delta_w < 0 (gap closes) and
the spread does not blow up (pooled sd at most 2x clean), use it. Otherwise fall back to the 15:08 fixed-sd
gap^2 loss and the post says why. The smoke numbers of both losses go to the receipts.
Limits text for the post (required): "The pre-registered attacker loss (batch d'^2) was replaced after a
20-step smoke test, before any scored run: it fooled the probe by adding noise (sd 2.08 -> 12.0) while the
class gap grew, and with that attacker a gap-based predictor has the wrong sign."

## 2026-10-17 15:11 CEST — disclosure: code smoke on a target (Bruno #1712)

A 10-step code smoke test of the save code ran on the TARGET dbpedia:NaturalPlace (lambda_KL 100, seed 0,
fixed-sd loss). Results deleted, never used; logged in data/smoke_tests.md. Seen: three SECONDARY probes
went 0.993 -> 0.635, 0.978 -> 0.626, 0.998 -> 0.499. No primary number was looked at. The scored run for
this target is a fresh run from the clean model.

## 2026-10-17 15:12 CEST — lambda_KL grid extended upward (pilot rule unbracketed; Kasper #1716)

The first pilot (fixed-sd loss) succeeded at the grid TOP, lambda_KL = 1000, at stop step 10 (AUROC
0.874 -> 0.565), so "largest grid value that still succeeds" picked the grid edge. The KL term has zero
gradient at the clean weights, so a 10-step attacker mostly follows the (Adam-preconditioned) probe gradient.
Fix, before any target run: the pilot grid is extended by x3 steps (3000, 10000, 30000, 1e5, ...) until one
value fails within the step cap; the rule then picks the largest succeeding value as written in PREREG.md.
Stop step and retain KL at stop are logged per run. If retain KL at stop is ~0 for all scored runs, the post
says the attacker stopped before the regime where the second-order law applies (PREREG "if not": early
stopping dominates). The pilot is re-run with the loss chosen by the 15:09 gate (softplus first).
- (15:13, Tove #1717) Added diagnostic, outside the verdict: per target, seed 0 only, the same attack with
  lambda_KL = 0 (5 extra runs). Report retain KL at stop for it, and KL(chosen lambda) / KL(lambda = 0) next
  to every R^2. A ratio near 1 means the retain loss did nothing, and the post says so (PREREG "if not").
  The three predictors are also scored on these lambda = 0 runs as an exploratory row.
- (15:14, Hugo #1719) F_common from 2000 windows has split-half correlation 0.88 only. Decision before any
  target run: the primary F is re-estimated from 8000 disjoint pile-10k odd-doc windows (2 sampled label
  draws per window), same estimator and damping rule; it runs as the 2nd heavy job next to the pilot and
  is frozen (hash in data/fisher/fisher_info.json) before the first target run finishes. The 2000-window F
  is kept as a sensitivity row. If 8000 windows do not exist or the job exceeds 15 min, use the largest
  count reached and say so.
- (15:18, Bruno #1731) F-8000 frozen before any target run finished: 8000 windows from 2303 odd docs,
  2 label draws, split-half corr 0.978 (was 0.880), median eig 0.03316 = lambda, 1 null eig (all-ones).
  sha256(F_common array bytes) = 88a09b34cc46563feb917181b689d96a7432285b71c42c1c98a92e5b14a740a4;
  sha256(data/fisher/fisher_big.npz) = 7cc9a3c7cc83db9c80b88a5a8e96c6ed9667e7b1b957c9af56cde82e62fc0634.

## 2026-10-17 15:41 CEST — pilot schedule (no rule change)

Softplus pilot failed at lambda_KL 1000 (AUROC 0.874 -> 0.789 at the 2000-step cap) and 300 (-> 0.769);
lambda 100 is running. The rule "largest succeeding grid value" does not depend on run order, so the
remaining values run in parallel on the 2 allowed slots: lambda 30 starts now next to 100; 10 next if both
fail. Time box: if no softplus value succeeds by 16:05 CEST, the attacker falls back to the 15:08 fixed-sd
gap^2 loss (bracketed upward from 1000), and the post says the published-style loss never pushed the
target's AUROC below 0.6 within 2000 steps at any lambda tried. Hugo #1734: at lambda 300-1000 the retain term
binds (no 10-step win), so the selected attacker is in the behaviour-preserving regime.
