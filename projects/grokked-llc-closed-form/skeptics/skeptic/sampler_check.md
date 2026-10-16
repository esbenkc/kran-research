# skeptic — sampler and calibration check (follow-up, G. Wang feedback, 2026-10-16)

Own code: `sampler_check.py` (generic model on an input matrix, own Jacobian, own SGLD loop, own chain
seeds, own Levenberg-Marquardt polish). Outputs: `sampler_check.json` (parts A-E), `sampler_check_F.json` (F).
Run: `uv run --no-project --with numpy python sampler_check.py A B C D E` and `OUT=sampler_check_F.json ... F`.

## Frozen setting (data/calibration.json, read from code)
Full-batch SGLD, own numpy (`qnet.sgld_llc`), not devinterp:
theta <- theta + eps/2 (-n beta grad L_n(theta) - gamma (theta - theta*)) + N(0, eps I);
lambda_hat = n beta (mean over steps 2,000..9,999 of L_n - L_n(theta*)), averaged over 4 chains.
L_n = (1/2N) ||P_perp (Yhat - Y)||_F^2 (centred over train pairs, no L2). n beta = 1e4 fixed. gamma = 1.
eps = 0.2 / (n beta h_max + gamma), h_max = top eigenvalue of J^T J / N at the polished point.
Checked: eps in llc.csv equals this rule on 53/53 rows (h_max from saved singular values).

## A. Calibration networks rebuilt from the stated recipe (teacher seed 0, W,V ~ N(0,1), targets = teacher outputs)
| net | N | truth (rank/2) | paper formula | mine est/true | build est/true |
|---|---|---|---|---|---|
| gauss_under_p5_K3 | 300 | 21.0 | 21.0 | +1.7% | +0.3% |
| gauss_under_p7_K6 | 400 | 60.0 | 60.0 | +1.1% | +0.0% |
| gauss_over_p3_K20 | 200 | 31.5 | 31.5 | +0.8% | +3.5% |
| onehot_p11_K6 | 60 | 93.0 | 96.0 | -0.9% | -1.5% |
| onehot_p17_K12 | 144 | 294.0 | 300.0 | -0.4% | -0.7% |
| onehot_p7_K14_cap | 24 | 80.5 | 140.0 | +1.3% | -0.3% |
| vonly_p11_K44 | 60 | 242.0 | 242.0 | +0.5% | +0.5% |
| vonly_p17_K68 | 144 | 578.0 | 578.0 | +0.9% | +1.0% |
Rank, h_min, eps identical to calibration_frozen.json on 8/8. All 8 have h_min >= 0.035 = 350x the
resolution gamma/(n beta) = 1e-4; 0 directions below it. "Fresh seed" = fresh sampler seed, same 8 teachers.
Wall time: 1-2 s per calibration net (4 x 10k steps).

## C. Grid (calibration_grid_main.json)
384 runs = 8 nets x scale {0.5,1,2} x crel {0.2,0.5} x gamma {1,10} x n beta {1e3,3e3,1e4,3e4}, 2 chains x 4k
steps, 0 diverged. Worst |rel err| per setting 3.8%-10.1%; frozen (0.2, 1, 1e4) = 3.87%, rank 4 of 16; top 5
within 3.82-3.90%. Not a strict argmin; the grid is flat because no calibration net has sloppy directions.

## D. Stationarity (data/stationarity*.csv): 11 of 14 checked cells settled (one checkpoint per cell,
2 chains x 40k steps, last fifth within 5% of the 10k reading). Unsettled: p7_K56 +14.9%, p31_K93 frac 0.5
+16.6%, p31_K124 frac 0.7 -5.2%. Rule recomputed from the csv: 14/14 agree.

## F. Finite-chain quadratic model (exact SGLD variance recursion on the saved spectra, 15 verdict nets)
Directions held only by gamma relax in ~2/(eps gamma) = 1-5 x 10^4 steps, but add ~0 loss. The loss-relevant
part: the 10k chain reaches 95-99% of its own equilibrium quadratic reading (p11 0.95-0.96, p17 0.95-0.98,
p23 0.97-0.98, p31 0.99). Under-equilibration of the quadratic part costs <= 5%.

## E. Sensitivity on grokked nets (my SGLD + my polish, 10k steps, 4 chains; ratio = lambda_hat / K(3p-1)/2)
| ckpt | gamma 0.1 / 1 / 10 / 100 | n beta 1e3 / 1e4 / 1e5 | crel 0.1 / 0.2 / 0.4 |
|---|---|---|---|
| p11_K33_s0 | 0.729 / 0.724 / 0.682 / 0.493 | 0.776 / 0.724 / 0.746 | 0.726 / 0.724 / 0.714 |
| p17_K51_s0 | 0.755 / 0.748 / 0.700 / 0.462 | 0.594 / 0.748 / 0.774 | 0.754 / 0.748 / 0.748 |
| p17_K68_s0 | 0.660 / 0.653 / 0.602 / 0.376 | 0.500 / 0.653 / 0.685 | 0.660 / 0.653 / 0.651 |
gamma 0.1 -> 1: < 1.1% change (plateau). gamma 10: -6 to -8%. gamma 100: -32 to -42%. Step size: <= 1.5%.
Max ratio over all 24 runs: 0.776 -> no tested setting puts these cells inside the 15% band. Wall time per
grokked net: 2 s (p11) to 15 s (p17), single thread.
