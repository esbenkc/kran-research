# Lemma: evasion transfer is set by the whitened cosine

Checks for every equation: `check_equations.py` (output in `check_equations.out`; E1–E6 below).

**Setup.** Two classes, `x | y ~ N(mu_y, Sigma)`, `Delta = mu_1 - mu_0`. A linear probe `u` scores
`s = u.x`.

**E1 (AUROC of a probe).** `s_1 - s_0 ~ N(u.Delta, 2 u'Sigma u)` for one random positive and one
random negative, so `AUROC_u = P(s_1 > s_0) = Phi(d'_u / sqrt 2)` with
`d'_u = u.Delta / sqrt(u'Sigma u)`. In words: `d'` is the gap between class means in units of the
probe's own score spread.

**E2 (the cheapest evasion).** The evader must lower `w.Delta` to a target `c`. If it pays the cost
`m' Sigma^-1 m` for a mean shift `m` (the KL divergence between `N(mu, Sigma)` and
`N(mu + m, Sigma)`, times 2), the Lagrangian `m' Sigma^-1 m - lambda (w.(Delta + m) - c)` has
stationary point `m = (lambda/2) Sigma w`; the constraint fixes `lambda`, so
`m* = (c - w.Delta)/(w'Sigma w) * Sigma w`. In words: the cheapest move goes along `Sigma w`, not
along `w`. In the Euclidean cost `m'm` it goes along `w` instead (E2c).

**E3 (the transfer rule).** Put `Delta' = Delta - t Sigma w`, `Sigma` unchanged. Then
`u.Delta' = u.Delta - t u'Sigma w` and `w.Delta' = w.Delta - t w'Sigma w`. Divide by the score
spreads: `d'_w` drops by `t sqrt(w'Sigma w)`, and `d'_u` drops by
`t u'Sigma w / sqrt(u'Sigma u) = [t sqrt(w'Sigma w)] * cos_Sigma(u, w)`. With `k` = fraction of
`d'_w` removed:

    d'_u(after) = d'_u - k d'_w cos_Sigma(u, w),   cos_Sigma(u,w) = u'Sigma w / sqrt(u'Sigma u * w'Sigma w)

**E4 (plain-words meaning).** `cos_Sigma(u, w)` is the correlation of the two probes' scores
within one class: `Cov(u.x, w.x) = u'Sigma w`.

**E5 (refit).** The best linear probe on the new data has `d'^2 = Delta' Sigma^-1 Delta'
= Delta' Sigma^-1 Delta - k(2-k) d'_w^2`. It reaches zero only if `w ∝ Sigma^-1 Delta` (the optimal
probe, where `d'_w` already equals the full separation) and `k = 1`. Any non-optimal monitor leaves
separation for a refit probe.

**E6 (diagnostic, other evader).** An evader that instead projects out `w`'s score,
`f(x) = x - Sigma w (w.x - b)/(w'Sigma w)`, also changes the covariance. At full removal
`d'_u(after) = (d'_u - c d'_w) / sqrt(1 - c^2)` with `c = cos_Sigma(u, w)`: same numerator, larger
for high `|c|`. The data tell the two evaders apart by the variance of `w`-scores after training.
