# Vera (skeptic) — own derivation, theory-1023-norm-rescale-free

Setup. h ∈ R^d (d = 1536) is block 28's output at the country token (readout 28, pre-norm). The model's next-word
distribution at that position is p(h) = softmax(W (g ⊙ N(h))), N(h) = h / sqrt(ms(h) + ε), ms(h) = |h|²/d, ε = 1e-6.
Block 28's output at a position feeds only that position's final norm (it is the last block), so p depends on h only through N(h).
Probe (frozen, panel.npz tok28): s(h) = (w/σ)·(h − μ) + b = a(h) + b0 with a(h) = v·h, v = w/σ, b0 = b − v·μ.
Post-norm probe (tok29) reads g ⊙ N(h).

## L1 Scale invariance
N(c h) = c h / sqrt(c² ms + ε) = N(h) · sqrt(ms+ε)/sqrt(ms + ε/c²). With ε = 0: N(ch) = sign(c) N(h). So c > 0 is free;
c < 0 flips N and is NOT free. Relative error for ε > 0: |1 − sqrt((ms+ε)/(ms+ε/c²))| ≤ ε|1/c² − 1| / (2 ms); ms ≈ 230²/1536 ≈ 34,
so ≈ 1.5e-8·|1/c²−1|. Numeric (vera_lemma.json): max KL over c ∈ {0.3, 0.55, 1.8, 3.0} = 1.4e-13; c = −1: 32 nats.

## L2 Euler (gradient orthogonal to h)
With ε = 0, KL(h) := KL(p_clean ‖ p(h)) satisfies KL(ch) = KL(h) for c > 0, i.e. it is homogeneous of degree 0.
Differentiate in c at c = 1: h · ∇KL(h) = 0. Holds at EVERY h, not only at clean. Numeric: |cos(∇KL, h)| ≤ 5.7e-10 at 8 perturbed points.

## L3 Null vector of the Hessian at clean
Differentiate h·∇KL(h) = 0 once more: ∇KL(h) + Hess(h) h = 0. At h = h_clean, ∇KL = 0 (minimum), so Hess h_clean = 0.
Numeric: radial curvature 5e-20 vs tangential 5e-5..1.6e-4.

## L4 Norm restore is output-neutral
h'' = h' |h|/|h'| is a positive rescale of h', so by L1 p(h'') = p(h') and N(h'') = N(h') (post-norm state and tok29 score unchanged).
Numeric: KL 4.5e-14; post-norm state relative change 1.2e-8.

## Title metric R (PREREG s6, Hugo #5430)
g(X) = mean over 596 eval twin pairs of s(false) − s(true twin) (b0 cancels). R = (g_restored − g_attacked)/(g_clean − g_attacked).
Decompose h' = c h + t, t ⊥ h, c = (h'·h)/|h|². Pure positive rescale (t = 0): restore gives h, R = 1. Pure turn with |h'| = |h|: restore is
identity, R = 0. Turn that also grows the norm (|t| ≠ 0, c = 1): restore shrinks by |h|/sqrt(|h|²+|t|²) = 1 − O(|t|²/|h|²): second order.

## First-order R of a state-space gradient step (step-0 prediction)
Probe loss gradient w.r.t. a false state ∝ +v (push s down: δ = −η v), true twin δ = +η v. To first order |h+δ| = |h| + ĥ·δ, so
h'' = h + δ − (ĥ·δ)ĥ + O(η²): the restore removes exactly the radial part. Per statement the gap change returned by the restore is
v·δ_rad = (ĥ·δ)(v·ĥ) = −η (v·ĥ)², the attack's change is −η |v|². Pooled on the mean gap over twins:
R0 = Σ_i (v·ĥ_i)² / (n |v|²) = mean cos²(v, ĥ_i). Numeric: 0.27% (finite steps η ∈ {1e-3, 0.1, 1}: 0.265%, 0.263%, 0.244%).
False median 0.51%, true median 0.004%.
Caveat: this is gradient flow on the state. The LoRA moves the state through the kernel J_θ J_θᵀ (J_θ = ∂h/∂θ), which need not be
isotropic, so R after LoRA steps is not predicted by this number.

## Radial share of the score drop (exact)
s(h') − s(h) = (c − 1) a(h) + v·t. The radial term is (c − 1)·a_i. Energy share |(c−1)h|²/|h'−h|² is NOT the score share: with
|h| ≈ 230 a 3% shrink is ~7 norm units (49 squared units) while v·h is only 7% of |v||h| for false statements.
