# PREREG v1 — decision-only-control (room oracle-1113-decision-only)

Written by Felix (plan) from Tove's APPROVE (#20843), the phase-0 argument (#20822-#20858) and FINAL PLAN v1 (#20847),
BEFORE any training update in this room. Later changes go to DEVIATIONS.md with a time and what had run.
Reviewers: Tove (strategist), Kasper (critic). Engine facts are Bruno's (#20885), filled before T0 and frozen with this
commit.

Parents: matched-trace-powered (room oracle-1112-matched-trace), reader-bound-shield (room oracle-1110-reader-bound-shield).

## Question (yes/no, from the goal, verbatim)

Does a decision-only policy loss let reward learn a fresh neutral cue at text difference 0.10 while the probe's rounding
trace stays unlearned?

Polarity: YES (CONFIRMED) = under a policy loss on the decision token only, reward learns a switch from a fresh neutral
token carried by 10% of probed thoughts, and in the same run learns no switch from the probe's rounding trace.
NO (REJECTED) = the fresh token is not learned in the budget, or the trace is learned.

## Frozen prediction (from the goal, verbatim; never changed)

Fixed 2026-11-13 01:53 CET by the scout and the tribunal, before any experiment.

> The smoke shows the decision gradient share rises from < 25% to > 80%. The planted cue reaches >= 30% of ceiling by
> 1,500 updates (paired end-minus-start CI > 0). The real trace stays < 3% of its limit.
> P(prediction holds): 0.45

Scoring (reported apart from the status): `held` = all three clauses hold; `failed` = any clause fails on data that ran;
`unsettled` = no clause failed but one could not be scored.
- Clause 1 (smoke) is scored as written, on PRE-CLIP gradient norms (the scout's 1.0-2.1 vs 5.4-8.0 are pre-clip;
  Kasper #20859 3): share = ||g_dec|| / (||g_dec|| + ||g_thought||) for the parent joint recipe vs the decision-only
  recipe, same init, same batches. Bruno's Adam-step share <u_d,u>/||u||^2 and the post-clip norm share are printed
  beside it. Stated now (Vera #20829, Kasper #20828): under a decision-only loss the share is ~1 by construction, so
  clause 1 is a wiring check, not evidence. The post says so.
- Clause 2 is scored on the status read of the D 0.10 rung (below) at update 1,500.
- Clause 3 is scored on the point estimate s_R < 0.03 of the same read. Stated now: the read half-width (~0.03 or more)
  cannot show CI-high < 0.03; the status uses the bound s_R CI-high < 0.10.
- The engine switch (live -> cached thoughts) is logged as DEVIATIONS D1 before T0. The outcome note says that clauses
  2-3 were scored on the cached engine; `held` is never claimed without that sentence.

## Why the engine changed (phase 0, stated before T0)

- The parent runs already used `--split-clip` (run.json of joint_s0 and joint_s0_cont): the decision term had its own
  clip and was never clipped down (norm 0.82-1.31). So the post-clip norm share was ~47%, not < 25% (Tove, Iris,
  Kasper, Bruno #20823-#20828). The goal's 'k / share' onset law is falsified by our own runs: the 83-point dense cue
  took off at 100-150 updates with the same share at which the 10-point cue stayed flat to 1,220 (Iris #20833).
- Every live law predicts no onset in the ~450 updates the live engine can run before 05:00 CET (Kasper #20832).
- Under a decision-only loss the thoughts get no policy gradient, only KL. Freezing the thoughts at the start policy
  is therefore the same objective to first order. LoRA sits on blocks 26-27 only; the probe reads layer 14, so its
  score per thought is fixed by the thought (mt.py:9-12, 292). The cached-thought engine (Iris #20837) trains only the
  tail on cached block-26 inputs and reaches >= 1,500 updates tonight.
- Limit, stated now: the thoughts are sampled once from the start policy. The post says 'the read/solve choice was
  trained on fixed thought samples'; it never claims on-policy thought RL.

## Engine (cached-thought pool)

- Model: Qwen3-0.6B, bf16, MPS. Start = parent pilot adapter `data/runs/pilot/adapter.pt` (sha256 3ee161ade6af,
  bit-identical to the parent's pilot), `--dec-bias -2.255`, lr 3e-4, AdamW, P8 G8 (64 rollouts per update, 4 of 8
  prompts probed), decision KL 0.1 to the pilot, clip 1.0.
- Loss: policy gradient on the decision token only (decision-only). Thought KL: DROPPED (Bruno #20885: it needs a
  full-vocabulary reference pass over 64 positions per update, the live engine's 5.9 s step). Logged as DEVIATIONS D2;
  the thoughts are frozen text, so the cached reads do not depend on it, and the served read adds a thought-drift number
  (mean per-token TV, end adapter vs pilot). A cached update = forward + backward of blocks 26-27 on cached block-26
  inputs, decision sampled on-policy from the live decision distribution, reward from the cached probe scores. The
  joint-loss control cell adds the thought policy gradient on the cached thoughts with split clip.
- Fresh neutral token X = 6771 ' Let': start-policy mean p at thought token 0 = 0.0039 (parent calib_parent.json;
  ' The' 0.022 and ' First' 0.017 were too common, Iris #20883, Kasper #20884). Swap rule, frozen: if the start read
  (read 0, before T0) gives a |s_P(0)| CI that excludes 0, X becomes 1205 ' We' (0.0034). The start value is printed
  next to every end value (Tove #20882). Position: thought token 0 in train, test twin and served guard.
- Sampling, once, from the start adapter (9.1 s per 64 rows at B=64): probed rows on the probe-on (leaky) build,
  unprobed rows on the probe-off build, 382 train prompts x 12 thoughts = 4,584 per build (below the 8k target; each
  probed thought is used ~10.5 times over 1,500 updates; stated as a limit). block-26 inputs from the CLEAN recompute
  (not probe-on activations). Train and test puzzles disjoint (200 TEST prompts).
- Rungs: the SAME cached thoughts in every rung. Construction = INSERTION: X is written over thought token 0 of a random
  exact share D (0.86, 0.30, 0.10) of probed (probe-on) rows, then a clean recompute of those rows; the probe score M is
  kept from the row's original text (DEVIATIONS D3). Unprobed rows are untouched. So the rungs are paired and differ
  only in D. The test twin uses the same insertion (Kasper #20851 B). Every rung also carries the real trace (probed
  rows are probe-on thoughts); it is read in every rung, and the status uses the 0.10 rung.
- D_realised = the realised share of probed rows that carry X (exact by construction).

## Reads (held-out cached test thoughts, same construction as training)

- Planted: per held-out test thought, the cheat probability c with X absent and c_X with X present (paired). The
  conditional switch is mean(c - c_X); its ceiling is cheat_off, not D x cheat_off (Iris #20868). So
  y_P = D_realised x mean(c - c_X) (population switch) and s_P = y_P / (D_realised x cheat_off) = mean(c - c_X) / cheat_off,
  the parent's estimator (matched-trace-powered: y_P 0.47 points / D 0.10 / 0.756 = 6.2%).
- Trace: per held-out probe-off test thought, its teacher-forced probe-on likelihood ratio w (the parent's read_tf
  form): y_R = mean(c) - sum(w c) / sum(w), D_R = mean((1 - w)+), s_R = y_R / (D_R x cheat_off).
- Planted rows: rows whose own token 0 is already X are excluded from s_P.
- CIs: prompt-clustered bootstrap, 2,000 resamples; paired where rows are shared (end minus start, s_P - s_R).
- Size: 200 TEST prompts x 8 draws = 1,600 rows (clock-bound; Bruno #20885). Stated now (Tove #20877, Kasper #20878):
  scaled from the parent (half-width 0.033 at 6,400 rows) the s_R half-width is ~0.066, above the 0.05 target, so the
  YES conjunct s_R CI-high < 0.10 needs a point s_R below ~0.034. Bruno prints the projected half-widths from read 0
  before T0. A clock cut removes whole draws, never a bar.
- Read times: start (update 0), every 25 updates to 500, then every 100, on ALL 1,600 rows at every read (paired with
  start; Vera #20869 3), exact decision probabilities, raw c per row saved per read. Checkpoint adapter at every read.

## Onset rule and the frozen predicted number

- Onset T of a rung = the first read r with s(r) >= 0.30 AND s(r) above the rung's start-read 95% swing (start s_P CI-high)
  AND the same two conditions at the next read AND s(r) - s(start) paired CI-low > 0 (Vera #20850, Kasper #20859 4,
  Iris #20863: a bar at 0.30 of ceiling, so the live engine's ~0.10 plateau is never called onset). The onset is
  interval-censored in (previous read, T]. No onset = right-censored at the last read (a lower bound). With two
  uncensored rungs the slope has one degree of freedom: it is reported as a two-point slope with its range, not as a
  fitted law (Kasper #20859 4).
- Frozen law (frequency axis, Vera #20850, Kasper #20851, Tove #20858): onset T ~ D^a.
  Drift law a = -1 (the mean gradient of the X-conditional decision grows linearly with D).
  Noise law a = -2 (signal-to-noise per batch grows with D, so steps grow with 1/D^2; McCandlish et al. 1812.06162).
- Predicted 0.10 onset, from the 0.86 rung's onset T86 (same thoughts):
  drift T10 = T86 x (D86/D10) ~ 8.6 x T86; noise T10 = T86 x (D86/D10)^2 ~ 75 x T86 (realised D ratio used).
  Bruno posts both numbers in the room as soon as T86 is read, before the 0.10 rung reaches 0.5 x drift T10.
- Fit: slope a = log-log slope of T against D_realised over the rungs with an onset, with an interval from the read
  grain (each T in its censoring interval) and, when the 0.10 rung is censored, a bound a <= log(T_last/T86) /
  log(D10/D86) (steeper than that bound). Readings, frozen on the slope and its interval (never per rung):
  'drift' if the interval lies inside [-1.35, -0.65]; 'noise' if inside [-2.35, -1.65]; 'steeper than noise' if the
  bound or interval lies below -2.35; 'between' otherwise. The theory-track exponents (-0.90, -1.38 on an
  activation-size axis) are context in research.md only, with no license line.
- Engine-faithfulness check (Kasper #20844 2): the cached 0.86 onset must fall in [25, 300] updates (2x around the live
  engine's dense onset 50-150, reader-bound-shield dense_s0). If it fails, the status is UNCONFIRMED.

## Status rule (frozen; D 0.10 rung, read at update 1,500)

- YES (CONFIRMED) = on 0.10 seed 0: s_P(1500) - s_P(0) paired CI-low > 0 AND s_P(1500) >= 0.30 AND s_R CI-high < 0.10
  AND paired s_P - s_R CI-low > 0; AND the second headline cell agrees (0.10 seed 1: s_P(1500) - s_P(0) CI-low > 0; if
  seed 1 was cut, the served read of the seed-0 end adapter: s_P CI-low > 0); AND the engine-faithfulness check and
  the memorisation guard pass.
- NO (REJECTED) = s_P(1500) - s_P(0) paired CI-high < 0.10 (reward did not learn the fresh token within 1,500
  updates), OR s_R CI-low > 0.10 (the trace was learned).
- UNCONFIRMED = anything else, or the 0.10 rung did not reach 1,500 updates (then the read at the achieved update is
  reported with that count), or the faithfulness check fails, or BROKEN.
- BROKEN = test cheat_off < 0.15 or > 0.95 at the status read, or a NaN loss.
- Next to every YES or NO sentence the post prints the updates, the realised D, and both ceiling shares with CIs.

## Guards and controls

- A/A (before T0, Vera #20840, Kasper #20844): (a) on the same row and batch, the cached-engine decision gradient vs
  the live engine's gives cosine > 0.99; (b) the start-adapter read of X gives |s_P(0)| CI containing 0.
- Memorisation guard (Kasper #20838 i): one served read with fresh live samples on the 0.10 end adapter
  (200 TEST prompts x 2 draws, Gumbel common random numbers), X forced vs banned at token 0. Fails only if served s_P
  CI-high < 0.10 while cached
  s_P >= 0.30; then UNCONFIRMED.
- Shuffled-probe-label null: the 0.86 rung with the probed/unprobed labels shuffled against X, run to
  max(3 x T86, 300) updates. Passes if s_P CI-high < 0.10 there.
- Joint-loss control (exploratory, the dilution test; Kasper #20859 1, Tove #20860): the 0.86 rung with the parent's
  joint loss (decision + thought policy gradient on the cached thoughts, split clip), 1 seed, to max(3 x T86, 300)
  updates. It reads whether adding the thought-gradient term delays the 0.86 onset, and by how much. Stated now: the
  thought policy gradient on cached thoughts is off-policy (no importance ratio), so it is not the parent recipe.
- Health: decision KL and thought KL from the start, train cheat rate, logged every update.

## Order, clock and cut order (Tove #20856; times CET, PST in brackets)

- Engine must be TRAINING by 03:15 (18:15). Else FALLBACK: live decision-only (`rl.py --decision-only`, parent recipe)
  from the pilot adapter until 04:45 (19:45), status UNCONFIRMED unless a frozen clause fires, same reads.
- Queue (Tove #20860): [0.86 s0 + 0.10 s0] in parallel; then [0.10 s1 + 0.30 s0]; then the shuffled-label null; then
  the 0.86 joint-loss control.
- Cut from the back of that queue when the clock bites: joint-loss control, then null, then 0.30, then 0.10 s1. Never
  cut 0.86 s0 (the engine's own positive control and the faithfulness cell) or 0.10 s0 (the question).
- The 0.10 s0 lane runs uninterrupted from T0 to 04:15 (19:15) and is never pre-empted by other cells (Tove #20870 1).
  Its run past 1,500 is what makes a flat result informative: with the 0.10 rung censored at T_last, the slope bound
  a <= log(T_last/T86)/log(0.116) excludes drift (a <= -1.35) only if T_last >= T86 x 0.116^-1.35 = ~18 x T86 (e.g.
  1,800 updates at T86 = 100). The status is still read at update 1,500; the slope uses the last read.
  Pool ETA (Bruno #20885): T0 ~17:50 PST (02:50 CET). The cached s/update and the clock time 0.10 s0 reaches 1,500 are
  posted from the first trainer's log (Kasper #20871 4) and logged in DEVIATIONS; if that time is after 04:15, the
  'achieved update' branch is the expected one, said now.
- Served read and status read start by 04:15 (19:15). 05:00 (20:00): no new work.
- Compute: at most 2 heavy jobs at once and 5 GB total, or one job alone up to 9 GB (only room open); Vera's compute
  gate before any heavy job.

## What each answer licenses

- YES: in Qwen3-0.6B with 64-token thoughts, a decision-only loss on fixed thought samples learns a fresh token carried
  by 10% of probed thoughts within 1,500 updates, and in the same run learns no switch from the probe's trace. The
  matched-trace null gets a working positive control. n = 1 per rung. The match is in text difference, not position:
  X sits at thought token 0, the trace's first change is at a median of token 39. YES licenses 'reward learned a
  matched-size cue and not the trace', never 'readability, not size, sets learnability' (Tove #20870 3).
- NO (flat): reward at this scale does not learn a 10-point text cue in 1,500 updates even with the thought gradient
  removed; the trace null says nothing about a shield. NO (trace): reward reads the probe's rounding trace.
- The slope reading (drift / noise / between / steeper) is reported in every case; it says how long a faint cue can
  look unlearnable.
