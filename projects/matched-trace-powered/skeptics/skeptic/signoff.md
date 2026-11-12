# Skeptic sign-off (Vera) — matched-trace-powered

**Verdict on the question: CONFIRMED (Yes), by the frozen status rule (PREREG v1.1 @ 8e82ff3c).** Question: does the
saved joint adapter stay blind to the probe's 64-token rounding trace after a powered read and a continuation that
takes the matched planted cue past onset? The trace's switch at the end adapter is 0.0% of its limit
[-3.2%, +3.4%]. The planted cue's switch is 6.2% [4.9%, 7.6%]. The paired gap is +6.2 [+2.5, +9.7] share points.
**Binding limit:** the end read does NOT show that reward grew the planted cue. Its share is not above the start
adapter (pilot, trained without the planted cue) or the 674 adapter. Prediction outcome: FAILED.

All numbers come from my own code in this folder, run on the build's raw files. I did not read or run the build's
read_tf.py, summarize_tf.py, check_smoke.py or figure code.

## Inputs

- `data/reads/raw/end_tf.npz`: end adapter `data/runs/joint_s0_cont/adapter.pt` (global update 1,220), 32 draws x 200
  TEST prompts = 6,400 rows, complete (end_tf.json `complete: true`). Plain passes on draws 0-7 (1,600 rows).
- `data/reads/raw/smoke_674_tf.npz` and the parent's `joint_s0_final.npz` / `start_test.npz` (branch
  swarm/oracle-1111-matched-trace).
- `data/runs/joint_s0/curve.jsonl` + `data/runs/joint_s0_cont/curve.jsonl`; `joint_s0_cont/run.json` (steps_done 546,
  stop 19:40:12).

## End read (vera_end.py -> end.vera.json; prompt-cluster bootstrap 2,000)

| quantity | value [95% CI] |
|---|---|
| trace switch y_R, full (self-normalised LR) | +0.003 pts [-0.23, +0.24] |
| text part / decision spill | -0.03 pts [-0.28, +0.23] / +0.03 pts [-0.02, +0.09] |
| trace text difference D_R | 0.097 [0.094, 0.100] |
| cheat rate without the probe (cheat_off) | 0.756 [0.747, 0.765] |
| **trace share s_R** | **+0.0005 [-0.032, +0.034]** (half-width 0.033) |
| planted switch y_P (D1 form) | +0.47 pts [+0.36, +0.59]; forced +4.73 pts |
| planted text difference D_P | 0.100 (pX_off mean 0.084, 0 capped rows) |
| **planted share s_P** | **+0.062 [+0.048, +0.077]** |
| **paired gap s_P - s_R** | **+0.062 [+0.025, +0.097]** |
| mean w | 0.999 [0.993, 1.005]; w in [0.26, 2.64] |

Frozen rule: onset met (parent 550/600) yes; s_P CI-low 0.048 > 0 yes; gap CI-low 0.025 > 0 yes; s_R CI-high
0.034 < 0.10 yes -> **CONFIRMED**. Stable over 6 bootstrap seeds: s_R hi 0.033-0.035, s_P lo 0.048-0.049, gap lo
0.0245-0.0258 (vera_extra.json `boot_seed_stability`). Draw halves: s_R +0.004 (draws 0-15) and -0.002 (16-31).

## Estimator checks

- Smoke on the 674 adapter reproduces the 11-11 read bit for bit on all 32 matched (prompt, seed) rows: toks_nb and
  toks_lb equal, c_nb, c_X, c_lb, c_nlb, lp_nb, lp_lb_nb max diff 0.0, pX_off 5.9e-5 (my own row match by key).
- Void rule (LR vs plain on draws 0-7, same rows): LR minus plain +0.09 pts [-1.06, +1.27] -> not void. Plain y_R
  -0.14 pts [-1.23, +0.96].
- Per-draw heterogeneity of y_R over 32 draws: chi2 33.9 on 31 dof, within chance.
- Parent npz, same LR form (vera_lr_parent.py): 674 adapter s_R 0.063 [-0.04, 0.17]; start -0.009 [-0.10, 0.08].
  The baseline-free form mean(c(1-w)) is off by 0.48 pts from noise in mean w and is NOT the estimator.

## Null tests (luck)

| cell | observed | null 95% (2,000) | p |
|---|---|---|---|
| y_R LR, w permuted within prompt | +0.003 pts | [-0.18, +0.25] | 0.73 |
| y_R plain (draws 0-7), label swap in each CRN pair | -0.14 pts | [-1.08, +1.06] | 0.81 |
| y_P D1, label swap per row | +0.47 pts | [-0.11, +0.12] | < 0.0005 |

## Positive control: was the planted cue LEARNED? (vera_extra.py -> vera_extra.json, vera_extra2.json)

Same 200 TEST prompts, paired by prompt. Start = pilot adapter (trained with no planted cue, per Kasper's check of
pilot/run.json), 800 rows, R4. 674 = parent end read, 1,600 rows. Trace forms at start/674 are text-only (no
c_lb_nb saved there); spill at the end is +0.03 pts, so this does not matter.

| | s_P | s_R |
|---|---|---|
| start (pilot) | 0.039 [-0.019, +0.093] | -0.009 [-0.104, +0.085] |
| 674 | 0.100 [0.055, 0.143] | 0.063 [-0.036, 0.167] |
| end (1,220) | 0.062 [0.049, 0.076] | 0.000 [-0.032, 0.035] |
| end - start | +0.023 [-0.030, +0.081] | +0.010 [-0.081, +0.099] |
| end - 674 | -0.037 [-0.083, +0.010] | -0.062 [-0.170, +0.040] |

Paired gap at start: +0.048 [-0.058, +0.152]; end gap minus start gap +0.014 [-0.089, +0.117].
So writing ' Need' first carries a small switch already in the pilot adapter. The end read cannot show that reward
grew it. Kasper (#19920) got the same numbers from his own code (+0.024 [-0.029, +0.082]; -0.037 [-0.083, +0.011]).
The evidence that the planted cue was learned at all rests on the parent: onset rule at 550/600 on dev, and the
11-11 TEST change 674 - start +0.37 pts [+0.01, +0.74] (D1 y_P). The continuation did not add to it.

## Exposure and dev curve (vera_curve.py -> vera_curve.json)

- Final update 1,220 (674 + steps_done 546). Cue mass 0.10 x 32 x 1,220 = 3,904; logged cum_P_tv_mass + 2,156.8
  matches 3.2 x update to 1e-4 at all 10 continuation checks.
- Trace mass (trapezoid of 32 x dev_D over checks, last dev_D 0.099 carried flat 1,174-1,220): 3,651 = 0.94x the cue.
- Planted dev forced switch over the continuation: mean 0.056 (10 checks, range -0.033..+0.098) vs 0.094 at
  550-650. Flat to slightly down; no take-off.
- KL / reward around the restart (fresh AdamW): global 724 kl 1.56, train_r 0.843 vs 650 kl 1.45, train_r 0.849.
  No jump beyond the parent's spread (50-update resolution only; no per-update rows logged).

## Prereg clauses (scored apart from the status)

- Phase A (paired gap excludes 0, trace < 1.5 pts at 674): not run (no slot) -> not scored.
- Planted >= 30% of ceiling by ~1,400: 0.062 at 1,220 -> **fails**. Iris's frozen band at mass 3,904: centre 0.76,
  80% [0.25, 1.0] -> far below.
- Trace < 1.5 points: +0.003 pts [-0.23, +0.24] -> holds.
- Probe-on cheating moves >= 5 points: c(leaky, banned) 0.506 (start) -> 0.755 (end, plain draws 0-7), +24.9 pts ->
  holds (overall cheating rose too: cheat_off 0.567 -> 0.756).
- Outcome: **FAILED** (a clause failed on a read that ran).

## Definition checks (what the post may and may not say)

- May say: at the end adapter (1,220 updates), the probe's trace carries no switch: 0.0% of its limit
  [-3.2%, +3.4%], with text difference 0.097 vs the planted cue's 0.100 and exposure 3,651 vs 3,904 marked
  thoughts. The planted cue carries 6.2% [4.9%, 7.6%] on the same scale, so the read can see a switch of that size.
- May NOT say: 'reward learned the planted cue during the continuation', 'take-off', 'existence proof that text
  difference does not set learnability', or 'exactly zero'. The planted share did not grow after 674 and is not
  above the start adapter at the end. 'Same text difference' at the end = 'about the same (0.097 vs 0.100)'.
- The 1.5-pt clause is on the population scale (~27% of the trace's own limit at 674); it scores only the prediction.
- n = 1 training seed. One model (Qwen3-0.6B), one task, 64-token thoughts.

## What I did not rebuild

- The RL itself and the sampler: all rebuilds use the build's sampled tokens, log-probs and cheat probabilities. The
  A/A smoke and the bit-exact 674 reproduction are the evidence that the new read code matches the old one.

## Final verdict

CONFIRMED by the frozen rule: s_R +0.0005 [-0.032, +0.034], s_P +0.062 [+0.048, +0.077], gap +0.062
[+0.025, +0.097], onset met. The trace side is solid (half-width 0.033, nulls at p 0.73 / 0.81). The positive-control
side shows a planted switch at the end but not one that reward grew in this run; the post must say so.
