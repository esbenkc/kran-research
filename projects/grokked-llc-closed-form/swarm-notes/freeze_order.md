# Order of calibration freeze and first grokked LLC estimates

Exported verbatim from the swarm room `theory-1015-grokked-llc-closed` (agent chat log), for the post's sampler fold.
Timestamps are Copenhagen time (CEST, UTC+2), 2026-10-15. File times in `data/` are Pacific time (UTC-7): `data/calibration.json` 06:00:38 PDT, first file in `data/llc_traces/` 06:13:03 PDT.

## #651 2026-10-15T15:00:44+02:00 plan

T2 SGLD estimator + calibration on known-lambda cases — FROZEN: data/calibration.json -> crel=0.2, gamma=1, nbeta=1e4, eps=crel/(nbeta*h_max+gamma) per ckpt (h_max from J^T J/n; V-block for V-only), steps 10000, burn 2000, 4 chains, call qnet.sgld_llc(..., nbeta=1e4, freeze_W=...). Fresh-seed validation on 8 known-lambda cases (3 Gaussian where Cor 4.2 is exact incl. over-param, 3 one-hot planted incl. cap, 2 V-only): worst |rel err| 3.5% -> band = 15% floor. Grid: data/calibration/calibration_grid_main.json (errors flat over scale 0.5-2 and nbeta 3e3-3e4). Code calibrate.py. Also equations.py (sympy+numeric checks E1-E4, all pass, output data/equations_check.json). Suggest llc.csv add lambda_quad = sum 1/2 h/(h+gamma/nbeta) from the ckpt spectrum (diagnoses flat directions).

## #652 2026-10-15T15:02:02+02:00 build

build: training sweep DONE (45 runs, data/ckpt/, 53 npz). Grok ckpts in all (p,K) cells except p7 K21; mem ckpts only in p7 (K21,28,56), p11 (K44 s1, K88), p17 K136, p31 K248. T3 runner ready: run_llc.py (waits for data/calibration.json keys crel,gamma,nbeta,steps,burn,chains). EARLY SIGNAL with smoke setting crel .2/gamma 10/nbeta 1000 (not frozen, just a test): p11 K33 grok: paper 528, rank/2 511.5 (=K(3p-2)/2 exactly), lambda_hat 284, and lambda_quad=sum 1/2 h/(h+gamma/nbeta) from the exact Gauss-Newton spectrum = 249. Mem V-only p11 K44: Thm 5.2 = 242 (exact for V-only), lambda_hat 160, quad 156. So the sampler is accurate; the gap is the localisation: grokked/mem Hessians have many eigenvalues h << gamma/nbeta=0.01 that a finite-nbeta SGLD does not count. Calibration on well-conditioned planted teachers will not see this. @plan: please include the h spectrum width in calibration, or report lambda_hat vs lambda_quad per ckpt (I log lambda_quad in llc.csv).

## #687 2026-10-15T15:13:06+02:00 build

build: run_llc.py STARTED 06:15 PDT with frozen calibration.json (crel .2, gamma 1, nbeta 1e4, 10k steps, burn 2k, 4 chains). 53 ckpts. Appends rows to data/llc.csv as they finish (small cells first, 8 workers; big cells p23 K184 / p31 K124,K248 last with 3 workers, Gram-eig spectrum). Extra cols: lambda_quad, n_resolved (h>gamma/nbeta), rank_method (svd tol 1e-9 / gram tol 1e-6), rank_gap. Spectrum per ckpt: data/llc_traces/<ckpt>.npz key singular_values (h = s^2/N), plus SGLD loss traces (full / pruned / vonly, thinned x10). Log: data/run_llc.log. ETA small cells ~15 min, all ~45-60 min (by ~07:15).
