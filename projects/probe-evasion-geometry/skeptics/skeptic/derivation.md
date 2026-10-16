# Skeptic's own derivation (written before reading build's code)

Setup. x | y ~ N(mu_y, Sigma), y in {0,1}, equal priors. dmu = mu_1 - mu_0.
Linear probe u scores s = u.x. Scores are Gaussian per class with the same variance u'Sigma u.
So AUROC(u) = Phi(d_u / sqrt 2), with d_u = u.dmu / sqrt(u'Sigma u) (the probe's SNR / d-prime).

Step 1 (smallest move). The evader shifts the class-1 mean by delta so probe w no longer separates:
w.(dmu + delta) = 0. Minimise delta' Sigma^-1 delta (Mahalanobis cost) subject to this.
Lagrange: 2 Sigma^-1 delta = lam w  =>  delta = (lam/2) Sigma w; constraint gives
delta = -(w.dmu / w'Sigma w) Sigma w.  [checked symbolically in derive_check.py, Claim A]
CAVEAT: the direction Sigma w comes from the Mahalanobis cost. With a Euclidean cost the move is
along w, and the transfer predictor becomes u.w / sqrt(u'Sigma u), NOT the whitened cosine.
Which cost the encoder "pays" is an assumption, not a theorem. The post must say so.

Step 2 (transfer). New d_u' = d_u - (w.dmu / w'Sigma w) * u'Sigma w / sqrt(u'Sigma u)
                          = d_u - d_w * cos_Sigma(w,u),
cos_Sigma(w,u) = u'Sigma w / sqrt(u'Sigma u * w'Sigma w) = cosine of Sigma^{1/2}w and Sigma^{1/2}u.
So the SNR drop is EXACTLY linear in the whitened cosine, with slope d_w. [Claim B]
AUROC drop = Phi(d_u/sqrt2) - Phi((d_u - d_w cos)/sqrt2): NOT linear in cos, and depends on d_u.
=> R^2 of AUROC drop vs cos alone is < 1 even when the lemma holds exactly; a probe family with
varied d_u adds variance the cosine cannot explain. I will report R^2 of (a) cos alone,
(b) the full closed-form predicted AUROC drop, and (c) raw cosine, and flag which one the
pre-registration means ("whitened-cosine prediction").

Step 3 (refit). After the move, dmu' = dmu - (w.dmu / w'Sigma w) Sigma w. A refit LDA probe has
SNR ||Sigma^{-1/2} dmu'||. It is 0 iff dmu' = 0 iff dmu is parallel to Sigma w iff w is the
LDA/Fisher direction Sigma^{-1} dmu. [Claim C]
=> If the fixed probe w is a well-trained logistic probe (near LDA), the lemma predicts full
collapse and refit AUROC ~ 0.5 under the minimal move. Refit recovery >= 0.9 then only happens
because the encoder does NOT make the minimal move (or w is far from LDA). This is a
consistency check on the pre-registered prediction: both halves cannot come from the same
minimal-move picture when w is near-optimal. Note: the covariance after the move is unchanged
under a pure mean shift; Sigma in the cosine should be the pre-move (= post-move) within-class
covariance. If build uses pooled covariance (including the class-mean term), that is a different
matrix: check which.
