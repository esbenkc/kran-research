# Skeptic sign-off (Vera) — optimizer-cue-size

Room theory-1112-optimizer-sets, 2026-11-12. Everything below was recomputed with my own code from the raw run JSONs
(`kran-research/projects/optimizer-cue-size/data/runs/*.json`), the raw cache (`raw/cache/eager_rec_tail.npy`) and the
Hugging Face Qwen3-0.6B weights. I did not read or run the builder's analysis code (analyze.py, pick_lr.py, common.py).

## Verdict on the question (headline, T5)

**SIGN-OFF: CONFIRMED** by the frozen rule (PREREG s6, my wording #20280-1). Pooled fit over the 24 headline runs
(3 optimizers x 4 cue sizes x 2 seeds, 0 censored, df 18, residual SD 0.123):

| | slope [95% t CI] |
|---|---|
| AdamW | -0.811 [-0.929, -0.693] |
| SGD (lr 0.01) | -1.484 [-1.602, -1.366] |
| sign-SGD | -0.996 [-1.114, -0.878] |

- d_SGD = -0.673, t CI [-0.840, -0.506], bootstrap [-0.768, -0.577]; the wider (t) decides. -0.673 <= -0.6 and
  CI-high < 0: YES clause 1. NO clause 1 (CI-low > -0.6) not met.
- d_sign = -0.185, t CI [-0.352, -0.018], bootstrap [-0.302, -0.067]; |d_sign| <= 0.3: YES clause 2. The CI
  intersects [-0.3, 0.3], so NO clause 2 is not met.
- Prereg outcome: **failed** on part (a) only: SGD slope -1.484 lies 0.016 outside the band [-2.5, -1.5] (its CI
  covers -1.5). Parts (b) and (c) held.

## Rebuild table

| Check | My number | Builder | Script |
|---|---|---|---|
| x per alpha, fp32 cue | 33.2638 / 16.6350 / 8.3176 / 4.1587 / 2.0793 | data/x_fp32.json, same to 4 decimals | vera_x.py |
| Phase 0 gate (cast cue, parent settings) | dev gap = parent at all 10 evals to 5 decimals; T 32 (s0), 24 (s1) | 32 / 24 | (room #20314) |
| SGD lr pick | lr 1-0.05 fail (P(X off) 0.000-0.004, entropy 0, top-1 1.00); 0.02 kept T 108; 0.01 kept T 180; pick 0.01 (abs ln 0.244 vs 0.267) | 0.01 | room #20346 |
| T for all runs (union of dev + dense curve, D >= 0.10 two evals in a row) | 0 mismatches vs T_dense over 33 runs | T_dense | vera_check.py |
| headline fit, contrasts, status | see table above; status YES | identical to 3 decimals | vera_check.py |
| switches are cue-conditional | at T: P(X off) 0.26-0.36, P(X on) 0.17-0.24 (SGD); D rises after T; health inside the D1 bounds in 24/24 | — | room #20357, vera_sens.py |

## Nulls and controls

- Freedman-Lane residual permutation (reduced model: own intercepts, common slope), 20,000 draws: d_SGD null 95%
  [-0.358, +0.356], observed -0.673, p < 5e-5 (0/20,000); d_sign null [-0.172, +0.172], observed -0.185, p 0.035.
- Exact within-size optimizer relabeling (1,296): null 95% [-1.89, +1.89], one-sided p 0.24. This null has almost no
  power here: SGD's T is 5-25x AdamW's at equal x, so a relabel mixes levels into slopes. Reported, not used.
- Shuffled-coin SGD control (penalty on an independent coin, x 33.3, 4,096 updates): no switch, max D 0.0133.
- Seeds are real: max |P(X off) s0 - s1| per cell 0.16-0.33.

## Fragility (the post must carry these)

1. Thin margin on a point bar: d_SGD -0.673 vs the -0.6 bar. Bootstrap P(d_SGD <= -0.6) = 0.91.
2. Switch level: 0.05 / 0.075 / 0.10 / 0.125 / 0.15 gives d_SGD -0.80 / -0.74 / -0.67 / -0.64 / -0.63; YES at all five.
3. Fixed start-up: subtracting c updates from every T gives d_SGD -0.62 (c 4), -0.56 (c 8, UNSETTLED), -0.47 (c 12).
   AdamW's T is 24-152, so its slope moves most (-0.81 -> -0.95 at c 8). The D4 time-matched AdamW arm (below) shows
   this correction is not licensed: the lr-free part of AdamW's T is 16 updates at x 33.3 but grows as the cue
   shrinks ((T_default - 16)/(T_tm - 16) = 0.09/0.12/0.13/0.13, not a constant 0.1), so a constant c is the wrong
   model; at SGD's time scale the gap stays (0.57 [0.42, 0.71]).
4. Drop one size: dropping x 33.3 gives d_SGD -0.58 (UNSETTLED); dropping any other size keeps YES.
5. SGD bends: quadratic term t -3.01 (p 0.03, 8 runs); local slope -1.69 over the two largest cues, -1.27 over the two
   smallest. AdamW is straight (-0.85 / -0.86). Over the two smallest cues d_SGD is about -0.41. The exponent gap is a
   property of this 0.9-decade range; the post must not extrapolate it to smaller cues.
6. sign-SGD is within 0.3 of AdamW by the frozen bar, but its CI excludes 0 (-0.185 [-0.352, -0.018]): the post may say
   "within 0.3 of AdamW", not "the same slope".

## Math

derive.py (sympy, asserts pass): drift with gap = x w and gradient ~ x gives SGD T = z/(eta c x^2) (slope -2), sign
T = z/(eta x) (slope -1), sign at low per-coordinate SNR back to slope -2 (erf expansion checked), halving eta doubles
SGD's T. Measured per-halving factors: AdamW x1.75, SGD x2.80, sign x1.99.

## Novelty

See novelty.md: the measurement is new as scoped; the mechanism is from 2402.19449 and 2512.00763; 2602.07729 (SGD
matches AdamW in LLM RL) is the closest RL claim, and this result is a boundary on it.
