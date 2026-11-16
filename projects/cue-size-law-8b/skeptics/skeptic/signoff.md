role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: cue-size-law-8b

SIGN-OFF: UNCONFIRMED (EXPLORATORY). The numbers reproduce; the status is capped by the frozen rule, not by doubt in
the numbers.

## What I rebuilt, with my own code, from the raw run JSONs only
Checker: `vera_final.py` (T = first eval with D >= 0.10 and D(next) >= 0.10 over every saved eval, D recomputed as
gap minus the step-0 gap; per-seed OLS of ln T on ln share; seed-level t-interval df n-1). Validated before any 8B
data on the parent's 13 sweep runs: pooled dense slope on x -0.90265 = parent summary dense.primary_x -0.902654.

| read | Qwen3-8B (mini, 3 seeds x 6 sizes) | Qwen3-0.6B matched arm (3 x 6) |
|---|---|---|
| dense T per seed (k0..k5) | s0 32/48/80/168/304/672; s1 32/48/88/160/328/672; s2 24/40/72/136/264/640 | s0 32/56/96/152/296/720; s1 24/48/80/152/304/664; s2 32/56/88/184/368/872 |
| per-seed slope | -0.886 / -0.890 / -0.936 | -0.867 / -0.939 / -0.944 |
| seed-level 95% (t, df 2) | -0.904 [-0.973, -0.835] = x1.87 [x1.79, x1.96] | -0.917 [-1.025, -0.809] |
| interpolated-crossing T | -0.968 [-1.044, -0.892] | -0.954 [-1.044, -0.864] |
| 8B minus 0.6B (Welch) | +0.013 [-0.076, +0.101] | |
| S2 calibrated constant | medians / prediction 0.80, 0.72, 0.77, 0.78, 0.92: all within 2x | |
| size-label permutation null | 95% [-0.47, +0.46], p = 0.0002 (4,000 draws) | 95% [-0.46, +0.47] |

Every saved dense T equals mine (36 main cells + 3 exploratory). All 36 main cells switched; |gap0| < 0.02 (8B k0
0.0192); step-0 P(X | off) 0.3435 in every 8B run (= G1 dev mean) and 0.3965 in every 0.6B run.

## Robustness (vera_robust.py, sensitivity only, not frozen)
- Drop any one size: 8B interval stays inside [-1.4, -0.7] in all 6 cases (-0.848 to -0.964).
- Curvature: ln T quadratic in ln share has c = +0.080 (t 4.0) at 8B and +0.072 (t 3.1) at 0.6B. Local slope -0.63 at
  the largest cue to -1.18 at the smallest, in both models. Top 3 sizes alone: 8B -0.73 [-0.89, -0.56], outside the
  band; bottom 3: -1.05 [-1.20, -0.90]. So x1.87 is an average over this range, not one power law.
- Start-up offset: ln(T - 8) gives -0.99, ln(T - 16) gives -1.12, in both models alike. The value of the exponent
  depends on the definition of T; the 8B-vs-0.6B equality does not (difference within +-0.07 in every subset).
- Unit: the absolute-time match holds in the frozen share unit (x / normed rows 0-2: 396 at 8B, 609 at 0.6B). In
  raw units (alpha, or absolute x) the 8B is ~1.5x smaller at the same alpha, so ~1.45x slower. The slope is unit-free.

## Gate and controls
- G2(b) failed as frozen: fp32 tail (training precision) on our cache vs NDIF 6/64 > 0.05, max 0.129 (data/gates.json);
  bf16 tail (NDIF's precision) on our cache 3/64, max 0.219; fp32 tail on NDIF's own input 2/64, max 0.140 (data/g2_diag.json). I checked it is not a port bug:
  bf16 tail on NDIF input max 0.027; per-token cache error median 0.0112, max 0.103 (my own read of
  raw/cache_qwen3-8b vs data/ndif); NDIF itself is deterministic only at a fixed batch size (batch 6 vs 64 moved rows
  1-2 by 1.1-1.8%, Iris). The run is EXPLORATORY (DEVIATIONS D2) and capped at UNCONFIRMED; correct per the frozen text.
- C1 shuffled coin, seed 0 (mini, full 8,192 updates): no switch. My rebuild: censored at 8,192, max D 0.0353 over all
  saved evals (the switch needs 0.10 twice). The run was killed at 09:22 and resumed from its update-4,096 checkpoint
  (DEVIATIONS D3, resumed_from [4096]); resume is statistically, not bitwise, the same run. C1 seed 1 and C2 wait in
  labq J1 (z9rh2wqu5x), so the frozen 'both C1 seeds' condition is not met tonight either.
- Null tests: size-label permutation (above, p 0.0002) and the shuffled coin (no switch) both say the slope needs
  the cue's size and the switch needs the cue.
- Partial seeds: the 0.6B extra seeds 3-4 were cut by the 10:15 stop. A seed with only the largest sizes sits on the
  flat end of the bend (0.6B seed 3 over k0-k3: -0.725), so it must not enter S1 next to full-grid seeds.

## Novelty
`novelty.md`: no published table of RL updates-to-switch against a cue's activation size at two model sizes. Souly et
al. 2510.07192 (near-constant poison count from 600M to 13B) is the nearest scale result, in supervised training.

## Verdict
PASS on the numbers: the key number (8B seed-level slope -0.904 [-0.973, -0.835]) and the 8B-0.6B difference
reproduce exactly from raw. Status UNCONFIRMED (EXPLORATORY) per the frozen rule (G2 failed; one of two C1 seeds run).
