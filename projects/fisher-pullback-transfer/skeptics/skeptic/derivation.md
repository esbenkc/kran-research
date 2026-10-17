# Vera (skeptic) — independent derivation

Written before seeing build's code or results. Checked numerically in `check_derivation.py`.

## 1. Retain cost is a pulled-back Fisher quadratic form

Edit `d` to the layer-L activation h. Output distribution p(y | h). KL(p(.|h) || p(.|h+d)).
First order vanishes (E_p[grad log p] = 0). Second order: KL ≈ (1/2) d' J' F_out J d,
F_out = diag(p) - p p' (softmax Fisher in logit space), J = d logits / d h.
So F = J' F_out J = E_{y~p}[g g'], g = d log p(y|h)/dh. (True Fisher: y sampled from the model.)

Common offset over T positions (same d added at every position, sequence log-lik summed):
KL_seq ≈ (1/2) d' (sum_{t,t'} F_{tt'}) d, F_{tt'} = E[g_t g_t'] where g_t = d log p(seq)/dh_t.
Equivalently g = sum_t g_t and F_seq = E[g g']. Per-token block-diagonal drops t≠t' terms.

## 2. Cheapest move against probe w

min_d (1/2) d' F d  s.t.  w'd = -δ  →  Lagrange: F d = -μ w → d* = -δ F^-1 w / (w'F^-1 w).
Cost at optimum: δ² / (2 w'F^-1 w). Damping: F_c = F + λ I.
Generic metric M (inverse of the cost matrix): d* ∝ M w, M ∈ {I, Σ, F^-1}.

## 3. Transfer to probe u (input-dependent edit)

Edit model: dh(x) = -(a + b s_w(x)) M w, s_w(x) = w'h(x) clean score.
Score of probe u after edit: s_u(x) - (a + b s_w(x)) u'Mw.
Class-mean gap on u's labels: Δg_u = -b (u'Mw) G_u, G_u = E[s_w | u+] - E[s_w | u-] = w'Δ_u.
Target: Δg_w = -b (w'Mw) G_w, G_w = w'Δ_w.
Eliminate b: Δg_u = Δg_w · (u'Mw / w'Mw) · (G_u / G_w).
With Δg_w = -κ_w w'Δ_w (κ_w = fractional loss of target gap):
  Δg_u = -κ_w (u'Mw / w'Mw) w'Δ_u.   (matches Felix #1648 (1))

Caveat found: for u = w the edit is an affine map of s_w, so the target AUROC either stays the same
(1 - b w'Mw > 0) or flips to 1 - AUROC (< 0); it never goes to 0.5 (check_derivation.py [3b] shows the flip
0.6252 -> 0.3748). So this edit model cannot itself produce the observed target drop to 0.6. It is used only
as a calibrated heuristic:
b is set from the measured target gap change; u's AUROC drop is predicted through the clean within-class sd
(Gaussian d' map) or, as sensitivity, exactly through empirical AUROC of s_u - c_u s_w.

## 4. Checks I run on raw data
- Recompute predictors from raw probe weights, Σ, F eigendecomposition, Δ_u, κ_w.
- Recompute measured drop from raw per-run AUROCs; R² = 1 - SSE/SST with held-out mean.
- Verdict: R²_F ≥ 0.6, R²_F - max(R²_Σ, R²_I) ≥ 0.10, R²_F - R²_shared ≥ 0.10.
- Nulls: random-direction F^-1 replacement (random PSD with matched spectrum), within-target permutation
  of predictor across probes, damping grid, seed spread, bootstrap over targets.
