# skeptic3 — novelty + pre-registered checks (2026-10-14 ~17:00 PDT)

## What is already known (verified from full text)
- Rosati et al. 2610.09004 (Prop. 3.2): gauge A_a = diag(aI, a^-1 I) acts on plain GD as
  preconditioner P = diag(a^-2 I, a^2 I). Paper text: "This identity holds throughout the trajectory for
  plain gradient descent; other optimizers and restricted trainable blocks require their own analysis."
  -> this sentence is the gap we test.
- Same paper ALREADY shows slowing under plain SGD: Deep Ignorance 6.9B unfiltered at a=1/24,
  median first 60% train acc 125 -> 225 updates (3 seeds, SGD lr 2.5e-4, 293 steps), and a Pythia-14M
  sweep a in {1/4..8} with SGD hitting times 865..1,825 steps (1 seed, lr 1e-5). So "gauge can slow SGD"
  is NOT new. BERT-Tiny 280 -> 25 is seed 0 only, a = 32 (speed-up case).
- Rosati et al. 2602.18868 (SpecDef): spectral reparameterization slows first- and second-order optimizers
  in the non-adversarial setting; Thm 8: informed attacker undoes it at linear parameter cost.
  They test Sophia, AdaHessian, Muon as curvature-aware attackers (Sophia "largely ineffective at
  undoing", Muon ~ Adam). They do NOT test K-FAC / natural gradient and do NOT test the Q/K gauge.
- WARP 2512.00272 (per skeptic): symmetry teleportation after unlearning, vs MIA, not relearning.

## What would be new
Q/K gauge (zero parameter cost, exactly invertible inside the architecture) on an unlearned causal LM,
against (a) AdamW, (b) K-FAC approximate natural gradient, (c) renormalize-then-attack.
Narrow but genuine. Frame: "first empirical check of the open question in 2610.09004 §6 for this gauge",
never "first to show attackers undo reparameterization defenses" (SpecDef owns that).

## Theory predictions (to compare with data, not to pre-commit the result)
- SGD: preconditioner diag(c^-2, c^2) on (Q, K). The QK-product update gets one term x c^2 and the other
  x c^-2. Away from c=1, one factor's effective lr GROWS -> at frozen lr expect instability/divergence,
  not a clean slowdown. A slowdown at frozen lr can be a mistuned-lr artifact (skeptic2 point 2 agrees).
- AdamW: step ~ lr per coordinate in gauged coords = lr/c on Q, lr*c on K in original coords. Same
  asymmetry: one factor freezes, the other gets x c. Expect speed-up or instability, not a robust slowdown.
- K-FAC (exact NG is invariant): K-FAC preconditions qkv with (A + lam I)^-1 (x) (G + lam I)^-1.
  The gauge scales Q-output grads by 1/c and K-output grads by c, so G blocks scale by 1/c^2 and c^2.
  The full-G inverse is invariant to this output rescaling ONLY when lam << eig(G). At c = 16, G_Q
  shrinks by 256 and FIXED damping lam dominates -> K-FAC on Q degrades to SGD with lr/lam.
  => K-FAC must use scale-relative damping (lam * trace(G)/dim, per factor), or it is not an invariant
  attacker and its line is meaningless. Log damping rule in meta.json.
  Also: G must be the full output covariance of the fused qkv Linear (or per-head blocks that contain
  both q and k rows of a head); a block-diagonal that splits q from k is still invariant (scalar per block),
  but a diagonal-only approx (e.g. EKFAC eigenbasis is fine, plain diagonal Fisher = Adam-like) is not.
- Renorm attacker: maps every c to the same weights -> flat by construction (skeptic2 point 4).

## Pre-registered skeptic3 sign-off conditions
1. meta.json written before c-grid runs: threshold, cap, lrs, damping rule (timestamps in git or file).
2. Gauge preservation: max|dlogit| < 1e-3 per c in fp32, and forget/retain acc at step 0 identical across c.
3. Pull-back null test for the attackers (my own code): for one step on a tiny batch, an exactly invariant
   optimizer gives theta'_1(c) = A_c theta_1(1). I check K-FAC and renorm implementations against this;
   SGD/AdamW must FAIL it (positive control).
4. Steps-to-recover recomputed from runs.jsonl with my own code; censored at cap stays censored;
   diverged (NaN/inf) is its own status.
5. Headline effect must exceed the c=1 seed spread (min..max over seeds).
