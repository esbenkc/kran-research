# skeptic2 sign-off — grokked-llc-closed-form

Claim under test (pre-registered, swarm plan.md): at grokked checkpoints the SGLD estimate of the
local learning coefficient matches the paper's closed form lambda = K_eff (3p-1)/2 (Cor 5.4 +
Remark 5.5 of Cullen et al., arXiv 2603.01192 v3) inside the calibrated band (15%) for >= 80% of
inside-hypothesis cells, with log-log slope in [0.85, 1.15]. Otherwise NULL.

## Verdict: SIGN-OFF on NULL

NULL holds: the SGLD reading at grokked checkpoints (p 11-23, inside the formula's range) is
0.65-0.78 of K_eff (3p-1)/2, outside the 15% band in every cell and every seed, with one sign.
It is also outside the band against my corrected count K_eff (3p-2)/2. The verdict does not
depend on how p31 is treated. Condition: the figure and summary.json must exclude p31 frac 0.5 (or
use the frac 0.7 rerun only if it settles), so that the figure title matches the post.

## 1. Math (my own derivation, derivation.md + derive_sympy.py)

- Shift symmetry: on one-hot inputs x = [e_a; e_b] each neuron has a second exact symmetry
  (u, s) -> (u + c 1, s - c 1). Sympy exact rank at p = 3: 7 = 3p-2. So the generic count is
  K(3p-2), not K(3p-1): the formula overcounts by K/2 (at most 1/(3p-1), i.e. <= 5% at p >= 7).
- Data cap: rank J <= p(N-1); the subabundant (inside-hypothesis) condition on one-hot data is
  K_eff (3p-2) < p(N-1). My recomputation of the inside flag from (p, K_eff, N) matches llc.csv on
  every row (verdict_recompute.json, 0 mismatches).
- L2 (smooth points): at a max-rank interpolating point lambda = rank/2 exactly. The grokked
  checkpoints reach full rank K(3p-2) only at SVD tolerance 1e-8..1e-10 (rank_tolerance.json), so
  "rank/2" is the value a sampler with unbounded resolution would read, not what SGLD reads.
- Memorising checkpoints with frozen first layer: the model is linear in V, so lambda = p rank(F~)/2
  by linear algebra; Thm 5.2 is an identity there. It is a check of the sampler, not of grokking.

## 2. Numerics (own code from raw data)

- verdict_recompute.py reads data/llc.csv + data/stationarity.csv only, recomputes lambda_paper,
  lambda_shift and the inside flag itself (does not use summary.json or run.py).
  Inside cells p 11-23 (5 cells, 12 checkpoints), ratio lambda_hat / lambda_paper:
  p11_K33 0.721, p17_K51 0.745, p17_K68 0.655, p23_K69 0.782, p23_K92 0.675.
  In band: 0/5. Every seed below 0.85. Same sign everywhere. Log-log slope 0.985.
  => NULL by the pre-registered rule. Also NULL against the corrected count K(3p-2)/2
  (every seed ratio <= 0.794).
- Robustness: adding the p31 train-fraction-0.5 cells (unsettled chains, room #700/#705) gives
  2 more cells (0.938, 0.825): in band 1/7, slope 1.079 -> still NULL. The verdict does not depend
  on how p31 is handled. The "share is constant across p" sentence DOES: it holds for p 11-23
  (medians 0.72 / 0.70 / 0.73); p31 must be shown separately or left out of that sentence.
- 40k-step chains (lead's stationarity.csv): every inside cell moves <= 3.4% from its 10k value;
  ratios at 40k 0.64-0.80. The shortfall is not a short-chain effect at p <= 23.
- Own SGLD rerun (compare_llc.py, rerun_sgld.jsonl, own polish, frozen setting): build's lambda_hat
  reproduced within 0.8% on 8 checkpoints (p7-p31).

## 3. Null tests (null_stepsize.py, gap_compare.py, dead_neurons.py)

- Step size crel 0.05-0.4: estimate flat within 1-2% -> not discretisation error.
- Regular quadratic with the checkpoint's exact Gauss-Newton spectrum (true LLC = rank/2 by
  construction) reads 0.76x / 0.80x of rank/2 at the frozen setting -> the shortfall size is what
  the curvature spectrum alone predicts, within 5-7%.
- Random point, same p, K, train set, weight scale: 0.04% of directions below the sampler
  resolution, estimate 0.96x of rank/2. Grokked point: 24% below, 0.76x. So the sampler is fine on
  generic points; grokked points are ill-conditioned (spectra span ~10 decades, no gap).
- Dead neurons add up to (2p-1)/4 each, visibility ~ nbeta/gamma^3; K_eff = K at all grok
  checkpoints, so this does not enter.

## 4. Novelty (novelty.md)

No prior quantitative test of Cor 5.4 / Remark 5.5 found; no prior statement of the shift
symmetry. Estimator sensitivity is known and the paper (v3) already reads absolute values "up to
an estimator-dependent scale". The post must not sell "absolute value is off" as refuting the
paper's scaling claim.

## 5. Post check

Checked blog.kran.ai/_notes/grokked-llc-closed-form.md (rewrite v5, 2026-10-15 ~23:12 CPH) line by
line against llc.csv, calibration_frozen.json, stationarity.csv, my derivation and my null tests.

Correct as written: per-neuron count 3p-1 (scale symmetry); the shift footnote (3p-2, at most 5%
lower; condition K_eff(3p-2) < p(N-1)); the SGLD estimator nbeta (E[L] - L(w*)); "half of MacKay's
effective number" for a quadratic loss (lambda_sur = 1/2 sum h/(h + gamma/nbeta)); 12 grokked nets
in 5 configs; 8 calibration nets, worst 3.5% (all their nonzero curvatures >= 0.03, far above the
1e-4 resolution: "had a gap" is right); 24% vs 0.04% below resolution at p17_K51 (my
gap_compare.json); step sizes over 8x move 1-2% (my null_stepsize.json); pure-Fourier count
(p-1)/2 (16p-24) = 3,784 vs 6,732 at p23, 99 neurons (my fourier_rank.json); NULL framed on the
absolute number, scaling with p, K stated as holding (slope 0.985 over 5 configs).

Fixed after my and skeptic3's notes (room #755, #762, #763): range 66 -> 65-78%; memorising points
now described as a sampler check (first layer frozen, model linear, formula exact); polish change
"at most 3% in grokked networks" (max 2.5%; 4.3% over all rows); "exact answer" -> "closed-form
answer". Open nit sent (#770): excerpt "the exact formula" -> "the paper's formula".

Still to check on the deploy preview: figure excludes p31 frac 0.5 and its title says ~0.72x;
caption [N] filled; equations render.

## 6. p31 train-fraction 0.7 rerun (lead, data/llc_p31_frac07.csv, first 2 seeds of K93)

hat/paper 0.873, 0.854 (inside the 15% band); noise-free lambda_quad/paper 0.808, 0.771, the same
range as p 11-23 (0.63-0.77). SGLD / lambda_quad rises with p at inside cells: p11 0.95-0.97,
p17 1.01-1.04, p23 1.02-1.06, p31 frac 0.7 1.08-1.11, p31 frac 0.5 1.26-1.48. So the higher p31
reading is the sampler sitting above its own curvature prediction, not a better-conditioned point.
Verdict: NULL either way (at most 2 of 7 cells in band). If p31 is shown, the post must not say
"same share at every p".

Update 23:20 CPH (verdict_recompute.py now also reads llc_p31_frac07.csv): p31 frac 0.7 K93 3 seeds
0.854-0.873 (lead: settled at 40k, -4.2%), K124 seed 0 0.757. Inside cells p 11-31 (7 configs):
ratio means 0.721 / 0.745 / 0.655 / 0.782 / 0.675 / 0.865 / 0.757, in band 1/7, slope 1.042,
one sign -> NULL. With p31 the range is 0.65-0.87, so "same share at every p" is not true;
"0.65-0.87, slope near 1" is.

## 7. Post v7 check (cold-reader rewrite, ~23:22 CPH)

New claims checked: half rank J is a lower bound (my L1); sampler 21-33% below it (ratio_rank
0.668-0.794); per-direction count 1/2 h/(h + gamma/nbeta) (re-derived: one Gaussian direction,
var 1/(nbeta h + gamma), nbeta E[L] = 1/2 nbeta h/(nbeta h + gamma)); calibration curvatures >= 0.035
or zero; steps to settle ~ nbeta at the resolution edge (from eps = crel/(nbeta h_max + gamma));
p(N-1) footnote; K_eff rule (|v||w|^2 >= 1% of max, qnet.py). Fixes sent (room #847): "formula =
infinitely fine sampler" needs the max-rank equality (my L2), not the lower bound; "2-7%" -> "at
most 7%"; polish "at most 3%" only for grokked nets (4.3% overall); open points 0.3-0.6; shift gap
1.5-3%; p31 sentences must follow the final p31 decision.

## 8. Final verdict set (lead #860), recomputed with my own code

Inside + settled: (11,33) 0.721, (17,51) 0.745, (17,68) 0.655, (23,69) 0.782, (23,92) 0.675,
(31,93, frac 0.7) 0.865; 15 networks; median 0.733; 1/6 in the 15% band; log-log slope 1.050;
vs half-rank bound 0.668-0.875 (12-33% below). Identical to data/summary.json. NULL. SIGN-OFF stands.
p31 K124 (frac 0.7) out: 40k change -5.2% breaks the pre-set 5% rule. Per skeptic3 (#858), p31 is
the only p where 10x finer resolution lowers the reading, so the post must not say the gap shrinks
at p31.
