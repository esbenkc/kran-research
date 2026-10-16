# skeptic3 — independent derivation

Model. Label y in {0,1}. Representation x | y ~ N(mu_y, Sigma), Sigma positive definite,
shared by both classes. delta = mu_1 - mu_0. A linear probe u scores s = u^T x.

## D1. AUROC of a linear probe
s | y ~ N(u^T mu_y, u^T Sigma u). AUROC = P(s_1 > s_0) for independent draws.
s_1 - s_0 ~ N(u^T delta, 2 u^T Sigma u). So

    AUROC(u) = Phi( d_u / sqrt(2) ),   d_u := u^T delta / sqrt(u^T Sigma u)   (signed d-prime)

Check: derive.py C1 (Monte Carlo, 4 random instances).

## D2. Smallest move that blinds probe w
Evader shifts the class-1 mean by Delta (class 0 fixed; any split of the shift between classes
gives the same delta' = delta + Delta). "Blind" = w^T delta' = 0 (AUROC(w) = 0.5).

The answer depends on the norm the evader pays:
- Mahalanobis cost  Delta^T Sigma^{-1} Delta  (= 2 x KL between the shifted and unshifted Gaussian):
  Lagrangian gives Delta* = -(w^T delta / w^T Sigma w) Sigma w.   (direction Sigma w)
- Euclidean cost ||Delta||^2: Delta* = -(w^T delta / w^T w) w.     (direction w)

Only the first gives the "moves along Sigma w" statement in the goal. "Smallest-norm" must be read
as smallest *Mahalanobis* norm. Check: derive.py C2 (sympy KKT + numeric minimiser for both costs).

## D3. Collateral damage on another probe u (Mahalanobis evader)
d'_u = u^T(delta + Delta*) / sqrt(u^T Sigma u)
     = d_u - (w^T delta / w^T Sigma w) (u^T Sigma w) / sqrt(u^T Sigma u)
     = d_u - d_w * cos_Sigma(w,u),     cos_Sigma(w,u) := w^T Sigma u / sqrt(w^T Sigma w * u^T Sigma u)

So the predicted AUROC drop is

    drop_u = Phi(d_u/sqrt2) - Phi((d_u - d_w cos_Sigma(w,u))/sqrt2)

Note: the drop is NOT a function of cos_Sigma alone; it also needs d_u and d_w. It is linear in
cos_Sigma only in the d-prime domain. For the Euclidean evader the analogue is
d'_u = d_u - (w^T delta / w^T w)(u^T w)/sqrt(u^T Sigma u) — a different quantity that equals the
whitened one iff Sigma is proportional to I (on the relevant subspace).
Check: derive.py C3 (sympy, generic 3x3 symbolic Sigma; plus Monte Carlo).

## D4. Refit after the Mahalanobis move
Best linear probe on new data is LDA, Sigma^{-1} delta', with d_opt'^2 = delta'^T Sigma^{-1} delta'.
Expanding:

    d_opt'^2 = D^2 - (w^T delta)^2 / (w^T Sigma w) = D^2 (1 - cos_Sigma(w, Sigma^{-1}delta)^2),   D^2 := delta^T Sigma^{-1} delta

(Cauchy-Schwarz in the Sigma inner product.) So the refit recovers AUROC Phi(D sin(theta)/sqrt2),
theta = whitened angle between w and the Bayes-optimal probe. Class means collapse (delta' = 0)
exactly when w is parallel to Sigma^{-1} delta, i.e. when the attacked probe IS the optimal probe.
Refit AUROC >= 0.9 needs D sin(theta) >= sqrt2 * Phi^{-1}(0.9) = 1.812.
Check: derive.py C4.

## Hinge assumption (main skeptic concern)
D2-D4 assume (i) a pure mean shift with Sigma unchanged, (ii) the evader minimises Mahalanobis
cost. A trained encoder is free to (i) change covariance / be non-Gaussian, (ii) pay whatever cost
its preservation loss implies. MSE-to-original preservation is Euclidean -> raw-cosine regime.
A test that "whitened beats raw" is only informative if Sigma is far from isotropic AND the
preservation loss is stated. derive.py C5 trains a linear encoder by gradient descent under both
losses to see which cosine wins.
