role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: theory-1124-llc-early-alarm-8b

Status I sign: UNCONFIRMED (the frozen Phase-0 gate failed; no 8B learning coefficient was scored).
Everything below is my own code, written from the PREREG and the raw data files, not from the build's scripts.

## 1. Before data: the frozen rule (null simulations)
- v2 as first written (LOO z = (a-b)/sigma, one-checkpoint alarm): P(control clause holds | pure noise) = 0.034
  (70 control points, iid) to 0.12 (AR(1) 0.8): 'No' by construction. `vera_power.py`.
- Two fixes (pair rule, sqrt(2)) then made the 2-seed control clause VACUOUS on the per-size grids: z = d/RMS(d) needs
  n >= 18 points for two consecutive z >= 3; the grids have 8/9/11 points. P(C1 quiet) = 1.0000 exactly.
  `vera_power_final.py`. Fixed by a 3rd C1 seed (Felix #29947).
- Frozen rule (3 C1 seeds): main false alarm before T-8 = 0.001 (iid) to 0.035 (rho 0.8); a real 4-sigma step in a
  control is caught in 84-85% per size; P(all 9 control tests quiet | noise) = 0.94 (iid) / 0.48 (rho 0.8).
  `vera_power_v3.py`.

## 2. The gate (PREREG (9)) — reproduces as FAIL
- From the build's raw per-draw losses (`data/gate/grid.json`): 200 rows, lambda-hat and batch-means SE rebuilt with
  0 mismatches; 0 of 20 settings pass; best c 0.3 gamma 10: 196.9 (-16.6%) / 276.5 (-11.4%); 0 of 100 repeats within
  10% at both ranks. `vera_gate_check.py`, `vera_gate_check.log`.
- Miss in units of the estimator's own error bar: all 35 finished rep-0 estimates miss Aoyagi by >= 5.09 batch-means SE
  (5 of 40 diverged); over all 175 finite repeats the minimum is 4.17 SE. `vera_bias_se.py`.
- Independent sampler, own data draw (`vera_gate.py`, Aoyagi in `vera_aoyagi.py`): the estimate rises with the step
  (gamma 1, rank 1: 69 / 173 / 244 / 302 at c 0.03 / 0.1 / 0.2 / 0.3); long walks (2 x 4,000 steps) give 237.8 / 235.7
  (rank 1) and 303.4 / 308.8 (rank 8). The code is right; the 100-step budget is short. `vera_gate_grid.log`.
- Rank gap (true 312 - 236 = 76) at the frozen budget: 62-109% of truth on the build's draw, 25-83% on mine.
  `vera_gap.log`.

## 3. Exploratory cells
- E2 at 8B (`vera_e2_check.py`, raw draws in `data/llc/e2/`): 6/6 lambda-hats rebuilt exactly. The walk's distance from
  w* equals the pure-noise distance sqrt(eps t d), d = 2,424,832, to ratio 0.9935-1.0006 at all 48 logged points. At
  c 0.033 the loss per unit noise (L / (eps t)) is flat (~25-39), the quadratic regime; at c 0.1 it jumps ~5x after
  the walk passes ~100 in distance (KL 0.8-1.2 nats). Readings 1.31 / 26.3 / 22.2 (u32) are not monotone in c.
- Independent 8B forward pass + SGLD (own Qwen3 tail `vera_tail.py`, checked against the parent's base dev P(X):
  0.34353 vs 0.343543; own sampler `vera_llc8b.py`): u32, c 0.033: 1.48 +- 0.36 (build 1.31 +- 0.19), own h_max 47.57
  (build 47.92). A 16-probe Hutchinson trace of the KL Hessian, 59.2 +- 10.9, predicts n beta x eps/2 x 63 x trace =
  1.29 +- 0.24 for a noise walk on a quadratic: the small-step reading is a curvature readout (`vera_trace_check.log`).
- E2b (`vera_e2b_check.py`): 12.082 (SE 0.53) = stored. 100-step windows: 27, 21, 14, then 9-16 with no rise to
  3,200 steps (Felix's 'still rising' prediction failed). Distance = a noise + spring walk with no loss term to
  0.9991-1.0007 at all 128 points: the loss term changes the distance by < 0.1%; whether ~12 is the LLC is untested.
- RL replay: this run's m8_k2_s5 equals the parent's at 13/13 dev evals (bit-identical), T_dense 72 = 72.

## 4. Novelty
See `novelty.md`. "Too few SGLD steps bias the LLC low" is known (Lau et al. 2308.12108 App. H); the new part is the
measured size at a budget an 8B LoRA study can afford, on the estimator the study would run.

## 5. Definitions not stretched
- Question, status and lab_status: UNCONFIRMED by PREREG (6).1 ('gate fails'). No 8B LLC enters a status claim.
- Prereg outcome 'failed' is right: its first clause (gate passes) is false.
- Post numbers I checked on preview-248 (2d68c44e): 35 scored / >= 5 error bars (min 5.09); 15% [13%, 17%] over 5
  sampler repeats; 25-109% gap; 20-fold; walk ratio range; budget and calibration tables; Aoyagi 236 / 312 and the
  case conditions. Corrections I raised and saw fixed: '1,600 steps' -> 400-1,000; 37 -> 35 estimates; 'gap roughly
  right' -> 25-109%; 'never felt the loss' and 'proportional to step' wording removed.
- Null tests: pure-noise simulations of the frozen detector (section 1); the gate failure holds on an independent
  data draw and sampler (section 2), so it is not luck of one seed.
