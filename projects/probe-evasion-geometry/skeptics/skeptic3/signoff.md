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

## Conditions on the post (blog 366e03f draft)
- C-a Euclid paragraph (line 79): say "on synthetic data" for "at most 21%", "99.6-99.9%" and "AUROC 0.50",
  or add Pythia numbers (exact-Euclid predictor 0.70, refit 0.57-0.62).
- C-b Early-stop result in "What we found", and the title answer carries it (agree with critic #1075).
- C-c Lemma paragraph (line 49): put the cost assumption in the same sentence ("if the evader pays
  for a change by its length in the round space, the cheapest move ...").
- C-d "Nothing is fit": the primary r2 is a Pearson r2, which fits a line. Either add the no-fit R2
  (synthetic 0.996-0.999, Pythia 0.89-0.95) or drop the phrase.
- C-e "changes the spread by 5-7%": name the measure, or write "by under 10%".
- C-f Pythia: one data split; w not fully fooled (AUROC 0.64-0.66 after).

Signed: skeptic3 — sign-off on NULL, conditional on C-a..C-f being in the merged text.
