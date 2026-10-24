# Sign-off — Hugo (skeptic2), theory-1024-curvature / curvature-cut-probe

Own code only (numpy/scipy/torch-CPU; no build script imported). Raw inputs: `data/runs/*/ckpt_*.npz`,
`acts_*.npy`, `data/fits/probes.npz`, `data/clean/z_post.npy`, `data/ref/split.json`, Qwen2.5-1.5B-Instruct
`model.embed_tokens.weight` + `model.norm.weight` from the HF safetensors.

## Verdict: CONFIRMED that the status is REJECTED (answer to the title: No)

| quantity (horizon 150, D8) | U full probe | P cut probe (k=128) | R random-cut (k=128) | file |
|---|---|---|---|---|
| clean line (2nd-highest of 200 refit true) | -0.344 | 0.495 | 0.838 | check_runs.py:44 |
| clean eval catch | 98.15% | 93.96% | 94.63% | out/check_*.json step 0 |
| seed 0 price (min KL at country token, catch <= 20%) | 0.00500 | 0.00531 | 0.00260 | out/check_{U0,P0,R0}.json |
| seed 1 price | 0.00586 | 0.00434 | 0.00278 | out/check_{U1,P1,R1}.json |
| arm price (D1: pooled catch <= 20%, seed-mean KL) | 0.00543 | 0.00483 | 0.00269 | out/status_H150.json |
| ratio to U | 1 | 0.889 | 0.495 | rebuild_status.py |
| city bootstrap 95% (2000 draws) | | [0.863, 0.916] | [0.477, 0.515] | out/status_H150.json |
| step-150 catch at re-set line (s0/s1) | 1.17/1.51% | 1.17/0.67% | 0.34/0.67% | check_runs.py |
| step-150 catch at CLEAN line (s0/s1) | 66.4/61.4% | 67.0/68.8% | 27.0/35.7% | check_runs.py |

PREREG s7 NO rule: P 0.00483 <= 1.5 x U 0.00543 = 0.00814 AND max-seed P 0.00531 <= 1.5 x max-seed U 0.00586 =
0.00879: both hold -> REJECTED. 2 seeds per arm (U s1 started before 18:20, so U = new-code s0+s1; D10).

Sensitivity (none reaches the YES bar of 3x):
- every seed pairing P/U: 1.063, 0.906, 0.870, 0.741; R/U: 0.520, 0.443, 0.556, 0.474.
- D10 slip-case U set (new s0, s1 + 1023 s1, s2): P/U 0.878, R/U 0.489, still REJECTED.
- first checkpoint (step 50) seed-mean KL: U 0.01176, P 0.00876, R 0.00502 (P/U 0.74x).
- all-token KL at step 150, seed mean: U 0.00489, P 0.00521 (1.07x), R 0.00389.

## Null / control tests
1. Re-pool (repool.py): ckpt scores = frozen probe applied to saved `acts_{0,150}` readout 29, max |diff|
   0.0000 on 1192 eval rows for all 6 runs; readout 29 equals clean `z_post` (median rel diff 0) and attacked
   states stay on the norm sphere (median |z/gamma| 39.19 at step 0 and 150).
2. Statement luck: city bootstrap CI above; the P vs R split (0.89 vs 0.50) is far outside statement noise.
3. Specificity (cross_catch.py, out/cross_catch.txt): each fine-tune hides only from its own probe. After P
   s0/s1 the full probe still catches 90.1/92.6% and R 68.8/71.8%; after R s0/s1 U catches 96.6/96.3%, P 90.1/89.4%;
   after U s0/s1 P keeps 36.1/29.0%, R falls to 6.7/5.2%.
4. 1023 U runs rebuilt exactly as a pipeline check: price (H=300) 0.00396, seeds 0.00374/0.00366/0.00449;
   H=150 0.00571.
5. Null probes (null_probes.py, out/null_probes.txt): 200 random unit directions and 20 shuffled-label refits,
   same 2nd-highest-of-200 line rule, catch median 0.8% clean and 0.8-1.0% on every attacked step-150 state
   (random p95 5.6-8.1%, shuffled max 2.5-5.2%): the line rule alone does not create or hide anything, so the
   arm probes' fall from 94-98% to 0.3-1.5% is a real hide. Positive control, a fresh full-space truth probe
   fit on the 'probe' split: 95.3% clean, 71.5/78.0% after the cut-probe runs, 87.4/86.9% after the random-cut
   runs, 0.3/0.5% after the full-probe runs (it is close to the full probe).

## Math (derivation_check.py, out/derivation_check.txt)
- Hessian of KL(p||softmax(l+Wx)) at x=0 = W^T(diag p - pp^T)W, gradient 0 (sympy, exact).
- min of x'Fx/2 s.t. u'x = -D is D^2/(2 u'F^+u) when u is in range(F) (Cauchy-Schwarz on F^{1/2}x); 0 when u has
  a ker F component (toy: exact min 6e-13, moving along the null coordinate).
- pair split: min_a a^2/(2g_f) + (D-a)^2/(2g_t) = D^2/(2(g_f+g_t)) at a = D g_f/(g_f+g_t) (sympy, exact).
- the small-move formula is NOT a lower bound for large moves: toy exact/formula 1.004, 1.000, 0.996, 0.955 at
  D = 1e-3..1 (Iris's exact P minima 0.88-0.91x of the formula agree in sign). Only the exact minimum or a
  certificate is a floor; the post should call the formula the small-move value.
- eps regularisation u'(F+eps I)^-1 u <= u'F^+u, so the eps floor sits slightly ABOVE the pinv floor (<= 1e-4
  relative at eps <= 1e-6 lambda_max).

## Power gate (pre-run, posted #6097 before any P/R step)
Own full-vocab float64 F_i (gate_hugo.py, streamed W), n = 24 random eval twin pairs (seed 2410): second-order
pair floor ratio P/U 3.15x (eps x0.01/x100: 3.16/2.86), R/U 0.067x, literal P_F w 1.06x. Measured: P/U 0.89x,
R/U 0.50x. The geometry predicted the ORDER R < U < P of the lowest possible price; the trained fine-tune
followed it only for R (downward) and not for P.
Sphere found-move (sphere_np.py / ellip_hugo.py) is an upper bound ~1.5x above Iris's certified sphere
minimum on the shared statement (1325, U: certified 5.04e-5, mine ~7.8e-5 interpolated); not used for any
headline number.

## Not rebuilt by me
- the 298-pair right-country check (prereg item 5) and the k* OOF selection fits.
- the prereg outcome is FAILED on item 3 alone (all-token KL P/U 1.07x < 3x) whatever item 5 gives.
