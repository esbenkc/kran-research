# skeptic verdict: SIGN-OFF on status NULL (conditions below)

Room theory-1016-probe-evasion, 2026-10-16. Own code only; build's code not read.
Inputs: `data/runs/*.npz` (X_train, y_train, E_train, X_test, y_test, E_test, w, probes).

## Derivation (derivation.md, derive_check.py)
- Min Mahalanobis-norm move that nulls probe w: delta = -(w.Delta / w'Sigma w) Sigma w (sympy Lagrange, exact).
- d'_u(after) = d'_u - k d'_w cos_Sigma(u,w) (exact identity, numeric).
- Refit d'^2 = Delta'Sigma^-1 Delta - k(2-k) d'_w^2; collapse iff w = LDA direction (|Delta'| = 2e-15).
- Euclidean move predictor (post footnote) re-derived in check_euclid_pred.py docstring. All match the post.

## Recomputed numbers (check.py; own rank AUROC, own Ledoit-Wolf Sigma from X_train, own refit)
Decision arm (Mahalanobis retain, D2: LAM=5, 16k steps), seeds 0/1/2:

| setting | whitened r2 | plain r2 | refit AUROC | shuffled-cos null p95 | no-fit R2 whitened |
|---|---|---|---|---|---|
| synthetic | 0.9989 / 0.9993 / 0.9987 (0.999) | 0.474 / 0.567 / 0.385 (0.475) | 0.971 / 0.989 / 0.988 | 0.084 / 0.063 / 0.113 | 0.997 / 0.999 / 0.999 |
| Pythia-70M L3 | 0.965 / 0.974 / 0.970 (0.969) | 0.341 / 0.335 / 0.230 (0.302) | 0.997 / 0.997 / 0.997 | 0.142 / 0.165 / 0.194 | 0.896 / 0.945 / 0.892 |

- Pre-registered rule (all 3 parts, both settings): synthetic plain r2 0.475 > 0.40 -> **prediction FAILED -> NULL**. Correct status.
- Nulls: shuffled cos_Sigma (2000 perms, keep d'_u): observed beats every permutation in all 6 runs.
  Random direction instead of w: p95 0.36-0.59 synthetic, 0.19-0.24 LM. Seeds: 3 distinct synthetic draws.
- Robustness: test-set Sigma 0.998 (syn) / 0.94-0.96 (LM); moment-based d' 0.999 / 0.998.
- Mean-gap shift: cosine with Sigma w = -1.00 in all 6 runs (with w: -0.42..-0.54 syn, -0.82 LM).
LM literal early-stop (check_lm_earlystop.out): whitened r2 0.796 / 0.773 / 0.875 (mean 0.81), plain 0.298 / 0.298 / 0.248,
refit 0.999, but no-fit R2 whitened only 0.49 / 0.56 / 0.57 (the slope is off). So on LM the early-stop evader keeps
the ranking but not the size of the drops.
- Euclidean arm: whitened r2 0.21/0.12/0.00, plain 0.53/0.42/0.31, exact-Euclid predictor 0.999/0.999/0.996,
  refit 0.50/0.51/0.50. LM Euclid (seed 0 only, n=1): whitened 0.007, exact-Euclid 0.70, refit 0.62.
- Design matches PREREG: d=64, eigenvalues follow 1/i (ratio 0.98-1.03), optimal d' 3.98-4.03, 25 logistic +
  25 mass-mean probes, subspace dims 2-63, w = train mass-mean (cos 1.000).

## Literal frozen stop rule (LAM=10, stop at first train AUROC_w <= 0.6), synthetic
whitened r2 0.691 / 0.573 / 0.260 (mean 0.508, FAILS 0.7); plain r2 0.719 / 0.622 / 0.723 (plain BEATS whitened);
refit 0.995 / 0.995 / 0.967; mean-gap shift cos with Sigma w only -0.86 / -0.73 / -0.64.
So the whitened law holds only for an evader trained until its retain cost is minimised (deviation D2).

## Conditions for this sign-off (text the post must carry)
1. The literal-stop-rule numbers above, in the results or limits, in one plain sentence: an evader stopped as soon
   as it fools w has not taken the cheapest move yet, and then the plain cosine predicts better.
2. Bailey et al.'s Gaussian detector scores ||Sigma^-1/2 (x - mu)||^2, the distance of an activation from the benign
   mean. It does not charge ||Sigma^-1/2 (f(x) - x)||^2. Write "uses the same metric", not "charges the same cost".
3. "changes the spread of the cloud by only 5-7%": my measure (Frobenius change of within-class covariance on test)
   gives 7.0 / 7.7 / 10.0%. State the measure, or write "under 10%".
4. LM: the 3 seeds share one data split and one w (AUROC_w 0.9961 in all three); only probe family and evader init
   change. Say so. LM Euclid arm is n=1.
5. LM whitened dots sit above the diagonal (no-fit R2 0.89-0.95): the prediction under-estimates real drops.
6. Novelty (novelty.md): credit Ying, Hase, Kriegeskorte 2606.19603 and Demontis 1809.02861 as the post does now; call
   it a lemma, not a new theorem.

If 1 or 2 is missing in the final post, treat this as REJECT.

## Confirmation (blog commit 4e0a686)
Conditions 1-6 checked in the committed post: early-stop numbers in "What we found" + fold (match my
recompute), Bailey "uses the same metric", "pays exactly" sentence gone, "under 10%", Pythia one split + one w,
under-prediction stated, no-fit R2 given, answer says "settled evader", "only because" removed.
SIGN-OFF stands: status NULL. Open, non-blocking: footnote [^proof] line 126 has a TAB where "\text" should be
(renders "ext{after}"); outcome_note could add that the frozen stop rule also failed the whitened part (51%).
