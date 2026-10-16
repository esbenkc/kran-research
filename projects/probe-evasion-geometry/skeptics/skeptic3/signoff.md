# skeptic3 — sign-off (2026-10-16)

## Verdict
**Prereg outcome: FAILED -> lab_status "NULL".** I sign off on NULL, and on the lemma as stated
with its cost assumption. Conditions for the post text at the end.

## 1. Math (own derivation: derivation.md, derive.py, out/derive_results.json)
| Claim in post | My check | Result |
|---|---|---|
| AUROC = Phi(d'_u/sqrt2) | C1 Monte Carlo, 4 instances, n=40k/class | max err 0.0033 (MC noise) |
| Cheapest move under round-space (Mahalanobis/KL) cost is along Sigma w; Euclidean cost -> along w | C2 sympy feasibility + SLSQP for both costs | PASS (err 2e-7 / 2e-8) |
| d'_u -> d'_u - k d'_w cos_Sigma(u,w) | C3 sympy, generic symmetric 3x3 Sigma | PASS (exact) |
| Refit d'^2 = D^2 - (w.delta)^2/(w'Sigma w) = D^2(1 - cos_Sigma(w, Sigma^-1 delta)^2); collapse iff w is the optimal probe | C4 sympy | PASS (exact). Partial k: D^2 - k(2-k) d'_w^2 (plan E5) agrees at k=1 |
| cos_Sigma = within-class correlation of the two probes' scores | by definition (Cov(u'x,w'x)=u'Sigma w) | PASS |
| KL of mean shift = 1/2 delta' Sigma^-1 delta (footnote kl) | standard Gaussian KL | PASS |
| Euclid-move predictor (footnote euclid) | hand derivation: Delta = -k (w.delta/|w|^2) w => change in d'_u = -k d'_w cos(u,w) |u| sqrt(w'Sigma w)/(|w| sqrt(u'Sigma u)) | PASS, matches footnote |

## 2. Numerics from build's raw npz, own code (recompute.py -> out/recompute_*.{json,txt})
Own Ledoit-Wolf pooled Sigma from X_train, own sklearn AUROC, own logistic refit (C=1).
Pearson r^2 (frozen primary), per seed:

| Arm | whitened r2 | plain-cos r2 | refit AUROC | no-fit R2 (whitened) |
|---|---|---|---|---|
| Synthetic, Mahalanobis (decision) | 0.999 / 0.999 / 0.999 | 0.474 / 0.567 / 0.385 (mean 0.475 > 0.40: FAIL) | 0.971 / 0.989 / 0.988 | 0.996-0.999 |
| Pythia-70M L3, Mahalanobis (decision) | 0.965 / 0.974 / 0.970 | 0.341 / 0.335 / 0.230 (mean 0.302: pass) | 0.997 x3 | 0.89-0.95 |
| Synthetic, literal frozen early stop | 0.691 / 0.573 / 0.259 (mean 0.51: FAIL) | 0.719 / 0.622 / 0.723 | 0.967-0.995 | 0.08 mean |
| Pythia, literal frozen early stop | 0.796 / 0.773 / 0.875 (mean 0.81) | 0.298 / 0.298 / 0.248 | 0.999 x3 | 0.54 mean (under-predicts); spread change 29% |
| Synthetic, Euclidean cost | 0.21 / 0.12 / 0.00 | 0.53 / 0.42 / 0.31 | 0.50-0.51 (collapse) | exact-Euclid predictor 0.999/0.999/0.996 |
| Pythia, Euclidean cost | 0.007 / 0.065 / 0.069 | ~0.005 | 0.57-0.62 | exact-Euclid predictor 0.70/0.75/0.65 (mean 0.70) |

Matches build #1063, plan #1069, skeptic #1065, skeptic2 #1064/#1074.

## 3. Null tests (mean over 3 seeds, decision arms)
- Shuffle cos_Sigma across the 50 probes, keep each d'_u (1000 perms): synthetic r2 0.024 (q95 0.086),
  Pythia 0.074 (q95 0.157); 0/1000 perms reach the observed r2 in every seed.
- cos = 1 for every probe ("strong probes drop more", no angle): synthetic 0.07, Pythia 0.23.
- Placebo: cos_Sigma against 200 random directions instead of w: synthetic 0.21, Pythia 0.07.
So the cosine carries the signal; baseline-strength spread alone does not.

## 4. Independent replication with my own evader (derive.py --c5; out/derive_results_c5.json)
Different design on purpose: d=16, spectrum 0.05-20, w = LDA plus noise, 1x64 GELU residual MLP,
penalty = squared class-mean gap of w-scores, 1500 Adam steps, 3 seeds per cost.
- Mahalanobis cost: whitened-formula r2 0.999-1.000; Euclid-formula 0.46-0.78; refit 0.98 = theory 0.98.
- Euclidean cost: Euclid-formula r2 0.999-1.000; whitened-formula 0.48-0.79.
The cost metric decides which cosine predicts. The fork replicates outside build's design.

## 5. Skeptic points (not changing the outcome)
1. The test is close to the lemma's own premise by construction: the evader is told to pay in
   the round-space metric. The non-trivial content is that a label-blind MLP lands on a near-pure
   mean shift along Sigma w (shift cos -1.00, spread change 7-10% by my pooled-LW measure on test;
   build reports 5-7% with its measure) and that real, non-Gaussian LM activations still give 0.97.
2. The early-stop arm is the realistic attacker (stops when w is fooled). There whitened r2 = 0.51
   and the move sits between w and Sigma w (shift cos -0.82 with w, -0.74 with Sigma w). My reading
   [INFERENCE]: early in training the penalty's gradient on f(x) points along w (plain Euclidean),
   and the Mahalanobis retain term only rotates the move toward Sigma w as it converges. So the
   relevant metric is set by the optimiser path, not only the stated cost. -> next-post idea.
3. Raw-clause failure is a property of the probe family (corr(raw, whitened cos) 0.67-0.87), as
   skeptic/skeptic2 showed before the run. It does not refute the lemma.
4. Pythia: the 3 seeds share one data split and one w (only probes and encoder init vary): n=1 split.
   The evader there leaves w at AUROC 0.64-0.66 (k ~0.85), so "blinded" is wrong for LM.

## Conditions on the post
Round 1 (blog 366e03f draft): C-a Euclid numbers scoped to synthetic; C-b early-stop in results and
answer; C-c cost assumption in the lemma sentence; C-d "nothing is fit" vs Pearson r2; C-e name the
spread measure; C-f Pythia one split, w not fully fooled. Status after the full draft: C-a done,
C-f split done.

Round 2 (full draft, room #1099), MUST-FIX:
1. "Our prediction failed only because the plain angle..." is false: under the literal frozen stop
   rule the whitened part fails too (51%). Drop "only". The title answer and outcome_note must say the
   law holds once the evader settles at its cheapest move (the longer, deviation-logged run).
2. "A set of monitors is only as diverse as the correlation between their scores" contradicts the
   Euclidean run (score correlation = whitened cosine explains <= 21% there). Scope or cut.
3. Bailey's Gaussian detector scores the Mahalanobis distance of x from the benign mean, not the
   size of the edit: same metric, not the same cost. "Pays exactly the cost we tested" overclaims.
4. = C-c (cost condition in the lemma sentence).
5. = C-d (no-fit R2: synthetic 0.996-0.999, Pythia 0.89-0.95, or reword).
6. Footnote [^proof]: "\text" became TAB + "ext" (broken MathJax).
7. Demontis et al. credit is no longer cited in the body; re-add; "Its use for evasion is new" ->
   "We found no prior use of it for evasion."
Minor: name the "15%" spread measure (mine: 7-8% pooled-LW Frobenius on Pythia); refit under
Euclidean cost "about 0.5-0.6" (my refit 0.50-0.62).

Signed: skeptic3 — sign-off on NULL for the derivation and the numbers. Sign-off on the post text is
conditional on round-2 items 1-7; final confirmation is recorded below when they land.

## Final confirmation (blog 24e37ea, 2026-10-16 ~15:35 CEST)
Round-2 items 1-7 all fixed: "only" gone and the answer/TLDR/outcome_note carry the settled-evader
condition and the frozen-rule 51%; the "only as diverse" line is replaced by "every evasion audit must
name the cost the evader pays"; Bailey's detector is "the same metric" (footnote kl); the cost condition
sits in the lemma sentence; no-fit R2 reported; [^proof] uses \text (no TAB); Demontis credited in the
body and "We found no prior use of it for evasion". All numbers in the body and folds match my
recompute (full arrays and public receipts). One minor, non-blocking: the Euclidean fold says
"On Pythia (two seeds) ... Euclidean one 70-72%"; with the third seed it is 3 seeds, whitened 1-7%,
Euclidean formula 65-75%.

**SIGNED: skeptic3 signs off on lab_status "NULL" and on the post text.**
Public-receipt gap (not a sign-off blocker): the synthetic early-stop and Euclidean receipts carry no
scores; their numbers were checked from the full local arrays.
