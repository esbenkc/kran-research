# Skeptic sign-off (Vera) — reader-bound-shield

**Verdict on the title: UNCONFIRMED (Not settled).** The frozen gate (planted cue, switch >= 5 points with CI-low > 0)
did not pass at update 100 or 200, and the frozen D3 rule stopped it at 200. So the real-probe arm never ran and the
title cannot be answered yes or no. Prereg outcome: unsettled.
**Question (title):** Does reward teach a probe-off switch only where a labelled reader of the same size can read the
probe's trace?

All numbers below come from my own code in this folder, run on the build's raw npz files. I did not read or run the
build's summarize/read code or Iris's reader code.

## Pre-T0 pricing (before any gate update)

| what | number | file |
|---|---|---|
| converged ceiling of the natural switch, planted cue at D = f 0.107, start cheat 0.50, decision beta 0.10 (the frozen fill cell) | 8.40 pts (seed-1, 2x length: 8.46) | vera_ceiling_fill.json |
| same, parent beta 0.25 | 6.46 pts | vera_ceiling_fill.json |
| real arm (no readable cue) probe-on move at the fill cell | +11.2 pts | vera_ceiling_fill.json |

## Gate (planted cue ' Need', seed 0), rebuilt

| look | y = switch, pts [95% CI] | forced switch, pts [CI] | D | f | cheat off / on | pair-swap null 95% | file |
|---|---|---|---|---|---|---|---|
| update 100 (dev trigger) | -0.05 [-1.52, +1.41] | +0.86 [-1.90, +3.57] | 0.261 | 0.256 | 0.576 / 0.576 | [-1.55, +1.52], p 0.94 | planted_s0_gate_100.vera.json |
| update 200 (fixed) | +0.11 [-0.34, +0.57] | -0.24 [-2.91, +2.35] | 0.052 | 0.040 | 0.728 / 0.727 | [-0.45, +0.42], p 0.63 | planted_s0_gate_200.vera.json |

Gate PASS needs y >= 5 with CI-low > 0: neither look passes. D3 (frozen before the 200 look) stopped the gate at 200
because the TEST forced switch was < +3. So the '500 updates' clause was not run.

## Speed check (is the gate failure the reward geometry or the engine?)

vera_speed.py calibrates my idealised engine model (free logit per thought type, the run's shaping) to the run's own
drift (off cheat 0.51 -> ~0.70-0.79 by update 50: logit step 2-4). At that speed the idealised learner shows a forced
switch of 58-75 pts at update 100 and 72-80 at update 200 (p10 >= 44, 20 seeds). The run showed +0.86 and -0.24. The
update moved the global cheat rate at full speed and moved the cue-conditional decision by ~0. [SUPERSEDED by the
exposure and dense results below: the idealised model needs far fewer cue examples than the real engine, which
learned the dense cue fully but only after ~1,400-3,400 cue-marked thoughts; the planted run stopped at ~820. So the
speed check shows the engine is slower per cue example than the idealised model, not that it cannot learn the cue.]

## Data count

data/parent/start_planted_train.npz: 1,528 CRN pairs, 152 differing (133 prompts); test 800 / 86. The reader fits on
~118 distinct cue pairs after its 20% validation split (x_planted_start_pairs.npz fit_diff_rows). [SUPERSEDED estimate
'~600-1,700' removed: the D-weighted count from vera_exposure.py is ~820 cue-marked thoughts by update 200.]

## Frozen rule, applied (vera_status.py)

- Status: UNCONFIRMED (gate did not pass; D3 stop at 200; no real arm ran).
- Prereg outcome: unsettled (gate clause stopped at 200 of 500; seed 1 and the real and bit-exact cells did not run).
  Clauses that ran: planted switch at 200 = +0.11 (< 5); probe-on cheat moved +22.8 pts (0.498 -> 0.727; start from
  the parent start TEST read with the frozen bias added to the decision logit).
## Dense-cue diagnostic (D3; same engine, ' Need' on ~85% of probed thoughts, 200 updates), rebuilt

| look | y, pts [CI] | forced, pts [CI] | D | cheat off / on | ceiling share [CI] | nulls 95% | file |
|---|---|---|---|---|---|---|---|
| update 200 | +82.76 [+80.27, +85.29] | +96.2 [+95.3, +97.0] | 0.864 | 0.957 / 0.130 | 1.000 [0.969, 1.033] | pair-swap [-6.1, +6.1], prompt-flip [-11.3, +12.2], p 0.000 | dense_s0_gate_200.vera.json |

Planted ceiling shares, same joint bootstrap: @100 -0.3% [-10.2%, +9.3%], @200 +3.0% [-8.9%, +14.9%]. So the engine
CAN learn a switch, up to its text-difference ceiling (y 82.8 <= D 86.4), from a cue on most watched thoughts.

## Cue exposure (why 'rare vs frequent' is not licensed)

Cue-marked thoughts = dev D x 32 probed rows per update, trapezoid over curve.jsonl dev_D (read every 50 updates,
so about +-30%): planted reached ~820 by its stop at 200. Dense: ~680 at update 25 (training gap +0.4), ~1,400-2,000
at updates 50-75 (first gap +3.1, +7.2), ~2,700-3,400 at 101-125 (take-off, +16.1), then +53.8 at 126-150. On
exposure, the planted run stopped before the point where the dense run began to move. Planted training gap per
25-update window: +2.0, +0.1, +4.5, +2.0, -0.5, +2.2, +3.1, -1.2 (SE ~2.5 each): noise.

## Dense seed 1 (DEVIATIONS D5, onset replication), rebuilt with vera_exposure.py

Training gap per 25-update window: +0.5, +5.5, +4.1, +3.1, +11.3. First window >= 7 = updates 101-125, at ~2,515-3,198
cue-marked thoughts (seed 0: ~1,359-2,036). Frozen rule (window start >= 1,000 and end <= 3,000): NOT replicated. The
onset is 'about 1,400 to 3,200 over two runs', still above the planted run's ~820 at its stop.

## Reader edge x (frozen rule, Iris's fits), rebuilt with vera_reader.py

| texts | x, pts [CI] | ceiling f | acc on differing | distinct marked fit pairs | null 95% | file |
|---|---|---|---|---|---|---|
| planted START (parent start texts, bias -2.255) | +10.12 [+8.00, +12.36] | 10.75 | 0.971 (86 pairs) | 118 | [-2.13, +2.14] | x_planted_start_pairs.vera.json |
| planted END (update-100 texts) | +25.62 [+22.50, +28.50] | 25.625 | 1.000 (205 pairs) | 344 | [-3.62, +3.38] | x_planted_end_pairs.vera.json |

Both: test rows = the read's differing pairs exactly; fit / val / test prompts disjoint (306 / 76 / 200).
Bound on the same update-100 texts: switch y -0.05 [-1.52, +1.41] <= reader x +25.6. y <= x holds but is not tight,
and it is untested where it matters (reader edge ~0 with text difference > 0: the real probe), because that arm did
not run.

Learning curves (val edge vs labelled looks, 8 per update): END texts +0.9 at 400 looks, +24.7 at 800; START texts
+0.5 at 400, +2.1 at 800, +10.3 at 1,200. Reward had ~460 cue-marked thoughts at update 100 and ~820 at 200. At
matched looks neither had learned much, so this run does NOT show 'labels beat reward per example'.

## Definition checks

- One unit: x and y are both mean over all test CRN pairs of P(read | probe-off text) - P(read | probe-on text),
  in points. Checked in my code (vera_read.py gap; vera_reader.py d over all 800 pairs).
- Ceiling: |y text part| <= D (TV); dense y 82.8 <= D 86.4. f >= D is the looser pairing count.
- 'Probe-on cheating moves >= 5' start value: parent start TEST read with the frozen bias on the 2-way decision
  logit (0.498), end 0.727 (+22.8) in the planted arm.
- Not stretched: the post must not claim 'reward ignores a rare cue' (matched updates) or a labels-vs-reward ratio
  (matched looks show none). Exposure counts are coarse (+-30%).

## Null tests

Every switch and reader cell has an in-pair label-swap null and (switches) a prompt-level sign-flip null, 2,000 draws
each (numbers in the tables). Planted switches sit inside their null bands (p 0.94, 0.63); dense switch and both
reader edges sit far outside (p 0.000).

## What I did not rebuild

- The RL itself (no independent sampler/trainer); all rebuilds use the build's sampled tokens and log-probs.
- Bit-exact and real arms: not run.

## Final verdict

UNCONFIRMED (Not settled). Gate failed: planted switch -0.05 [-1.52, +1.41] at 100 and +0.11 [-0.34, +0.57] at 200,
forced switch +0.86 and -0.24. The engine can learn: dense cue +82.8 [+80.3, +85.3] at its ceiling (D 86.4). Reader
edges +10.1 (start texts) and +25.6 (update-100 texts). The real-probe question stays open.
