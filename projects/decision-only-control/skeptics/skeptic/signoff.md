# Skeptic sign-off (Vera) — decision-only-control

**Verdict on the question: CONFIRMED (Yes), by the frozen status rule (PREREG v1 @ f394f3a3 + DEVIATIONS V1.1-V1.10).**
Question: does a decision-only policy loss let reward learn a fresh neutral cue at text difference 0.10 while the
probe's rounding trace stays unlearned? On the 0.10 run at update 1,500 the planted word ' We' reaches 86.5% of its
limit [85.6, 87.4] (start 0.6%), and the probe's trace sits at -0.4% of its limit [-7.6, +6.6]. The paired gap is
+86.9 points [+79.7, +94.1]. Fresh live samples on the end adapter give 80.8% [77.7, 83.6].

**Binding limits (must stay in the post):**
1. The YES holds for a decision-only loss with group-mean advantages WITHOUT the std division and the probe penalty
   moved 0.40 -> 0.42. At the planned recipe the same loss collapsed to 'always cheat' within 25 updates, for the
   0.10 cue (all 5,600 test rows at c = 1.000) and for the 0.86 cue (min c 0.986 at 25, 0.999 at 50).
2. Thoughts were fixed samples from the start policy (no on-policy thought RL); thought KL dropped (D2).
3. The match is in text difference (TV 0.10), not in position: ' We' sits at thought token 0, the trace is spread.
4. The trace drifts up late: mean over reads 1,100-1,500 minus mean over reads 0-500 = +2.4 points [-0.1, +4.8].
   Not significant, but 'not learned in 1,500 updates (2.5x the planted onset)', never 'unlearnable'.
5. n = 1 per strength (0.86, 0.10), one model (Qwen3-0.6B), 64-token thoughts.

All numbers come from my own code in this folder on the build's raw files. I did not read or run the build's
cached.py, summ.py, served.py or figure code.

## Inputs

- `data/pool/test/cols.npz` (2,800 probe-off test thoughts + 2,800 X twins, 200 TEST prompts x 14 draws).
- `data/runs/<run>/reads/c_<step>.npy` for d010_s0 (0-1,500), d086_s0 (0-300), d010_s0_frozen, d086_frozen, null086_s0.
- `data/pool/on*/cols.npz` (training pools), `data/reads/served_d010_s0_1500.npz`.

## Construction checks (vera_check.py -> vera_check.json)

- Twins = plain rows with token 0 = 1205 (' We') and tokens 1-63 equal: 2,800 / 2,800. Rows whose own token 0 is
  already ' We': 12 (excluded from s_P).
- Training pools: realised D 0.0999 (0.10), 0.8599 (0.86), 0.300 (0.30, not run); marked rows carry ' We' at token 0,
  all other tokens equal the probe-on pool; unmarked rows identical. Natural ' We' at token 0 in unmarked probed
  rows 0.34-0.39%. Off-pool counts of the candidates reproduce Bruno #20896 exactly (' Let' 0.28% / 14.7%, ' We'
  0.33% / 1.4%).
- Start policy from the test pool's own d2_ref reproduces read 0 on MPS (max diff 3e-8): cheat_off 0.499,
  forced switch +0.0028 [+0.0014, +0.0042], s_P(0) +0.0056 [+0.0028, +0.0084] (CI excludes 0: A/A (b) fails as
  written; DEVIATIONS V1.3 handles it; the post must say 0.6%, not zero), D_R 0.1009, s_R(0) -0.024 [-0.080, +0.030].

## Headline cells (prompt-cluster bootstrap 2,000, every ratio recomputed per resample, paired with read 0)

| cell | value [95% CI] |
|---|---|
| d010_s0 s_P(1500) | +0.865 [+0.856, +0.874] |
| d010_s0 s_P(1500) - s_P(0) | +0.860 [+0.851, +0.868] |
| d010_s0 s_R(1500) (trace) | -0.004 [-0.076, +0.066] |
| d010_s0 s_P - s_R | +0.869 [+0.797, +0.941] |
| d010_s0 cheat_off(1500) | 0.511 |
| served, fresh live samples, 400 rows (vera_served.py) | s_P +0.808 [+0.777, +0.836]; 92% of rows switch the right way |
| d086_s0 s_P(300) / minus start | +0.925 [+0.919, +0.931] / +0.920 [+0.913, +0.925] (onset 100) |

Frozen rule: dP CI-low > 0 yes; s_P >= 0.30 yes; s_R CI-high 0.066 < 0.10 yes; gap CI-low > 0 yes; second cell
(served s_P CI-low 0.777 > 0) yes; memorisation guard passes; health (live probed groups >= 97.5%, train probed
cheat <= 0.56) not BROKEN -> **CONFIRMED**.

## Onset and slope

- d086_s0: s_P 0.007 / 0.015 / 0.091 / 0.613 / 0.759 at 25-125: onset T86 = 100, interval (75, 100].
- d010_s0: s_P 0.006-0.015 to 400, 0.044 (450), 0.039 (500), 0.402 (600), 0.666 (700): onset T10 = 600, (500, 600].
- Frozen forecasts posted before d010_s0 ran: drift (645, 860], noise (5,546, 7,395]. Observed 600: just before
  the drift window, far from noise. Two-point slope over the censoring corners -0.75 to -0.96 (point -0.83):
  frozen reading 'drift'. Marked thoughts at onset (32 probed rows per update): ~2,060-2,750 (0.86) vs
  ~1,600-1,920 (0.10).

## Null tests (luck)

| null | observed | null 95% | p |
|---|---|---|---|
| d010_s0 dP at 1,500, start/end read labels swapped per prompt (2,000) | +0.860 | [-0.126, +0.115] | < 0.0005 |
| d010_s0 dP at 700 | +0.660 | [-0.097, +0.090] | < 0.0005 |
| d086_s0 dP at 125 | +0.753 | [-0.104, +0.094] | < 0.0005 |
| d010_s0 trace at 1,500, w permuted within prompt (500) | -0.004 | [-0.057, +0.068] | 0.91 |
| served s_P, c_x <-> c_notx swapped per row (2,000) | +0.808 | [-0.175, +0.144] | < 0.0005 |
| shuffled-probe-label run null086_s0 (X on 86% of probed AND unprobed rows), update 300 = 3 x T86 | s_P +0.009 [+0.006, +0.012]; minus start +0.003 [+0.002, +0.005] | pass bar: CI-high < 0.10 | PASS |
| X-free training pool screen_nostd, update 100 | s_P minus start +0.004 [+0.002, +0.005] | — | baseline drift |

null086 and screen_nostd show a tiny probe-unrelated drift of ~0.3-0.4 points in s_P. The real 0.86 rung at the same
update (300) is at 92.5%, and the real 0.10 rung reaches 86.5% at 1,500. The take-off needs the X-probe link.

## Prereg outcome (reported apart from the status)

- Clause 2 (planted >= 30% by 1,500, paired CI > 0): held (86.5%).
- Clause 3 (trace < 3% of its limit, point): held on the point (-0.4%); CI-high 6.6% (the room's status bar is 10%).
- Clause 1 (smoke share < 25% -> > 80%): not run as written; scored only if joint086_frozen's pre-clip norms land.
  It is a wiring check (decision-only share = 1 by construction). Outcome as of this sign-off: unsettled.

## Novelty

See novelty.md: the frequency exponent of learning time is known in supervised settings (time ~ 1/p); we found no
paper that measures it for a reward-trained decision on a sparse text cue, or that pairs a matched-TV neutral cue
with a probe's own text trace as a positive control. New relative to our own posts: matched-trace-powered's control
failed (6.2%, not above start); here the control works (86.5%, +86.0 above start).
