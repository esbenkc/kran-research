# The transfer predictor: derivation

Every step below has a check in `check_equations.py` (output in `check_equations.out`, 16/16 pass).
Check IDs are in brackets. Prior art: the pull-back metric and the least-KL move (step 2) are derived by
FishBack (Wang, Zhao, Cao, Yao, Shu, arXiv 2605.17231) for activation steering. What is new here is only
step 3-4: using that move as a model of a real fine-tuning attacker and testing which other probes it fools.

## Setup

`h` in R^512 is the residual stream of Pythia-70M after block 2 (`hidden_states[3]`), mean-pooled over
the positions of a text window. A linear probe `u` reads the score `s_u = u'h`. The attacker fine-tunes
blocks 0-2, so it changes `h`; it is penalised by the KL divergence between the clean and the new
next-token distributions (the "retain loss"), and rewarded for making probe `w` (the target) fail.

## Step 1. The retain loss is a quadratic form in the edit [E1, E1b, E1c, E2, E2b]

In words: for a small edit, the KL penalty grows like the square of the edit, weighted by how much each
direction of the edit changes the model's output.

For one position with logits `z(h)`, output `p = softmax(z)`, and Jacobian `J = dz/dh`:

    KL(p(h) || p(h + d)) = 1/2 d' F d + O(|d|^3),   F = J' (diag p - p p') J.

`diag p - p p'` is the Fisher information of the softmax with respect to the logits; `F` is that matrix
"pulled back" to layer 3 through the later layers. The same `F` is the expected outer product of the
gradient of `log p(y)` with `y` SAMPLED from the model (true Fisher) [E1b]; with data labels instead one
gets a different matrix (empirical Fisher) [E1c].

For a window of `T` positions and one offset `d` added at every position, outputs at `s` depend on
`h_1..h_s`, so the cost is `1/2 d' (sum_{t,t'} F_tt') d`. Its sampled estimator is `E[g g']` with
`g = sum_t d/dh_t sum_s log p(y_s)` [E2, E2b]. The per-position sum `sum_t F_tt` drops the cross terms and
underestimates the cost by 12% in the toy check (KL ratio 1.135) [E2]. The experiment uses `F = E[g g'] / T` (the `1/T` only rescales).

## Step 2. The cheapest move against one probe [E3, E3b]

In words: among all edits that lower the target score by a fixed amount, the one with the smallest KL
cost points along `F^-1 w`, not along `w`.

    minimise 1/2 d'F d  subject to  w'd = -eps   =>   d* = -eps F^-1 w / (w' F^-1 w).

`F` is close to singular (LayerNorm makes the all-ones direction an exact null), so the experiment uses
`F + lambda I` with `lambda` = the median eigenvalue of `F`. This rule does not depend on the overall
scale of `F` [E10].

## Step 3. What the cheapest move does to another probe [E4, E4b]

    u' d* = -eps * r_P(u, w),   r_P(u, w) = u'P w / w'P w,   P = (F + lambda I)^-1.

In words: probe `u`'s score moves by the target's move times one ratio. The ratio does not change if `P`
is multiplied by a constant [E4b]. The same formula with `P = Sigma` (activation covariance) is the
"whitened" predictor; with `P = I` it is the "plain" predictor.

## Step 4. From a score move to an AUROC drop on different data [E5, E5b, E6]

The 50 probes are different concepts on different texts. AUROC ignores any shift that is the same for all
inputs, so the move must depend on the input. The simplest model: each input moves along `P w` by an
amount affine in its clean target score,

    dh(x) = -(a + b s_w(x)) P w.

Then probe `u`'s new score is exactly [E5]

    s_u(x) -> s_u(x) - a u'P w - kappa_w r_P(u, w) s_w(x),   kappa_w = -delta_w / G_w,

where `G_w` is the target's clean class-mean gap and `delta_w` its measured change after the attack, so
`kappa_w` is the fraction of the target's gap that the attack removed. The constant `a u'P w` does not
change any AUROC. The predicted AUROC after the attack is the AUROC of this new score on `u`'s own clean
test texts; it equals the AUROC of the actually edited activations in a known-answer test [E5b].

Caveat, stated in the post: on the target's own texts this affine edit only rescales `s_w`, so it cannot
by itself explain the target's own AUROC drop (unless it flips the sign) [E6]. `delta_w` serves only to set
the size `b`; the claim under test is that the part of the edit that is linear in the target score, along
`P w`, is what carries over to other probes.

## Step 5. Link to the 2026-10-16 result [E7, E8]

If `u` and `w` share the same labels and `P = Sigma`, the gap change of `u` in units of its sd is
`-k d'_w cos_Sigma(u, w)` with `k = kappa_w` [E7]: the whitened law of the previous post. The ratio form
and the literal cosine `cos_P(u, w) = u'Pw / sqrt(u'Pu w'Pw)` differ by the factor
`sqrt(u'Pu / u'Sigma u) / sqrt(w'Pw / w'Sigma w)` [E8]; they agree only when that factor is the same for
every probe.

## Background fact [E9]

For two classes with equal-variance Gaussian scores, `AUROC = Phi(d'/sqrt 2)`, `d'` = gap / sd. Used only
in the Gaussian sensitivity row.
