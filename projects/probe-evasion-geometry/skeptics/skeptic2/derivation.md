# skeptic2 — own derivation (written before any decision-run result)

Checks: `derive.py` → `derive_out.txt` (all PASS).

**Setup.** Two classes, `x | y ~ N(mu_y, Sigma)`, gap `delta = mu_1 - mu_0`. Linear probe `v`, score `v'x`.
Separation `a_v = v'delta / sqrt(2 v'Sigma v)`; the post's `d'_v = sqrt2 a_v`.

1. **AUROC.** The score gap between a random positive and a random negative is
   `N(v'delta, 2 v'Sigma v)`; AUROC = P(gap > 0) = `Phi(a_v)`. (C1, Monte Carlo.)
2. **Cheapest move.** Minimise `e' Sigma^-1 e` subject to `w'(delta + e) = 0`. Lagrange:
   `Sigma^-1 e = lam w` → `e = lam Sigma w`; the constraint gives
   `e = -(w'delta / w'Sigma w) Sigma w`. Strictly convex objective + affine constraint → unique. (C2.)
   In the Euclidean norm the same steps give `e = -(w'delta/w'w) w`. (C3.) The two differ unless `w`
   is an eigenvector of `Sigma`. **The direction is set by the evader's cost metric, not by the probe.**
3. **Transfer identity.** Take a fraction `t` of that move. Then
   `u'(delta + t e) = u'delta - t (w'delta)(u'Sigma w)/(w'Sigma w)`; divide by `sqrt(2 u'Sigma u)`:
   `a_u' = a_u - t a_w cos_Sigma(u,w)`, exact, `Sigma` unchanged. (C4 symbolic + n=50 numeric, C6.)
   With `k := t`, this is the post's `d'_u - k d'_w cos_Sigma`. ✔ matches PREREG / plan's lemma.md.
4. **Refit.** Best linear separation is `D = sqrt(delta' Sigma^-1 delta)`. After the move,
   `D'^2 = D^2 - 2t (w'delta)^2/(w'Sigma w) + t^2 (w'delta)^2/(w'Sigma w) = D^2 - k(2-k) d'_w^2`. ✔ matches
   PREREG. At `k=1`: `D' = D sin(theta)`, `cos theta = d'_w / D` (whitened angle between `w` and
   `Sigma^-1 delta`). Collapse iff `w` is optimal. (C5.) This is the single-probe corner of LEACE's
   theorem (equal class means ⇔ no linear probe beats chance).
5. **Duality.** For a cost metric `A` (cost `e'Ae`), the move is along `A^-1 w` and full collapse
   happens iff `A^-1 w ∥ delta`. Mahalanobis + LDA probe, or Euclidean + mass-mean probe, both collapse.
   Verified in `oracle_evader.py` (Euclid arm, mass-mean `w`: refit 0.44-0.50).
6. **What a real evader can do.** The encoder sees `x`, not `y`. Under the frozen per-sample loss
   `E[(f(x)-x)' Sigma^-1 (f(x)-x)]` with the w-gap constraint, the pointwise Lagrangian gives
   `f(x) = x + gamma Sigma w (pi(x) - p)` with `pi(x) = P(y=1|x)`. Mean gaps follow item 3 exactly;
   the within-class covariance changes by `Var(pi | y)`, which the lemma ignores. A linear encoder
   (`pi` replaced by its linear projection) shrinks the w-score variance to `w'Sigma w (1 - cos^2 theta)`.
   (C7 for the Euclidean linear case.) Size of the effect on the prereg r^2: `oracle_evader.py`
   (synthetic, r^2 0.996-0.998 at d*=4; ~0.93 at d*=2) and `oracle_lm.py` (LM: 0.92-0.95).

**Points where the post must be careful.**
- "Smallest move" must name its norm (Mahalanobis = KL cost between equal-covariance Gaussians).
- The AUROC drop is not a function of `cos_Sigma` alone; the predictor uses `d'_u`, `d'_w`, `k`.
- The PREREG "raw baseline" (plain cosine in the same formula) is not the exact Euclidean-move
  predictor; that one carries the factors `(|u|/sqrt(u'Sigma u))(sqrt(w'Sigma w)/|w|)`.
- In this synthetic design raw r^2 ≈ 0.3-0.65 even when the lemma is exact (corr of the two
  cosines 0.64-0.90), so a miss of "raw <= 0.4" there is a design property.
