# DEVIATIONS — decision-only-control

Each entry: time (PST), what had run when it was made, what changed, who asked.

## D1 — engine: live sampler -> cached-thought pool (2026-11-12 ~17:10 PST, before T0)

What had run: nothing heavy in this room (code only).

The goal's design is one live joint Qwen3-0.6B run of ~1,500 updates with a decision-only policy loss. Phase 0 found:
the parent already used split clip (post-clip norm share ~47%, not < 25%); the live engine runs 17.4-17.5 s/update, so
only ~450 updates fit before 05:00 CET, where every live onset law predicts a flat cue (~0 bits; Kasper #20832). Under
a decision-only loss the thoughts get no policy gradient, so freezing them at the start policy is the same objective to
first order. Tove approved the cached-thought engine and the D ladder 0.86 / 0.30 / 0.10 (#20843, #20856-#20860).
The goal's frozen prediction is unchanged; clauses 2-3 are scored on the cached engine and the outcome note says so
(PREREG.md, scoring).

## D2 — no thought KL (before T0; Bruno #20885, Vera #20872)

The goal's design gives the thoughts a KL term. In the cached engine the thoughts are frozen text, and the thought KL
needs a full-vocabulary reference pass over 64 positions per update (the live engine's 5.9 s step). It is dropped.
The cached reads do not depend on it. The served guard on the 0.10 end adapter adds a thought-drift number (mean
per-token TV of end-adapter vs pilot thoughts on the served prompts).

## D3 — probe score of inserted rows (before T0; Bruno #20885)

When X is written over thought token 0 of a probed row, the row's probe score M is kept from its original text (the
probe reads layer 14 at the answer token; one changed token is not re-scored). The reward of an X row therefore does
not depend on X itself, only on the row being probed.

## V1.1 — faithfulness band and 'seed 1' defined (2026-11-12 17:11 PST, before T0; Kasper #20889)

What had run: pool sampling only (no training update, no read).
- (A) Faithfulness: only the HIGH end is a status gate: the cached decision-only 0.86 onset must be <= 300 updates.
  An onset below 25 is a reported flag (a faster decision-only onset is the effect under test, not unfaithfulness).
  When the cached joint-loss 0.86 cell runs, its onset is reported against the live joint-loss band [25, 300].
- (B) 'Seed 1' in the cached engine = the same pool, a new random insertion subset (which probed rows carry X at the
  same exact share D) and a new seed for prompt order and decision sampling.

## V1.2 — test pool size and X swap screen (2026-11-12 17:12 PST, before T0; Bruno #20893, Tove #20877, Kasper #20878)

What had run: pool sampling only (no training update, no read).
- Test pool: 200 TEST prompts x 14 draws = 2,800 rows (was 8 draws, 1,600 rows). Projected s_R half-width ~0.05.
  Every read uses all 2,800 rows.
- 'Thought token 0' = the first sampled thought token (index 0 of toks, the slot where rl.py's planted cue acts). Train
  rungs, test twin and served guard all write X there.
- X swap screen, added before the rungs are built: X = 6771 ' Let' becomes 1205 ' We' if ' Let' sits at index 0 in
  more than 1% of off-pool thoughts, OR the read-0 |s_P(0)| CI excludes 0. Bruno prints ' Let' counts at index 0 and
  anywhere in the off and on pools.

## V1.3 — X = 1205 ' We' (2026-11-12 17:25 PST, before T0; Bruno #20896, Kasper #20884)

What had run: the off pool only (4,584 probe-off thoughts; it stores an unused x1 column). No rung, test pool,
training update or read existed.
Off-pool counts, first thought token / anywhere in the thought: ' Let' 0.28% / 14.7%; ' We' 0.33% / 1.4%; ' The'
2.4% / 13.8%; ' First' 1.6% / 6.6%. ' Let' passes the index-0 screen but is a common word inside thoughts; ' We' has
the same index-0 rate and is 10x rarer anywhere. X = 1205 ' We'. Its start forced switch comes from read 0 on all
2,800 test rows; the read-0 swap rule now has no fallback token: if |s_P(0)| CI excludes 0, the post reports it and
uses end-minus-start only.

## V1.4 — saturation at the frozen recipe; lr 1e-4; new health rule; race reading (2026-11-12 18:08 PST; Tove #20915)

What had run: run d010_s0 at the frozen recipe (lr 3e-4, decision KL 0.1), killed after 25 updates
(logs/d010_s0.log, data/runs/d010_s0/reads.jsonl). Read 0: cheat_off 0.499, s_P 0.0056, s_R -0.024. Read 25: test
cheat_off 1.000, c_X 1.000, s_P 0.000, s_R 0.000; train cheat 0.96, live groups 13%, decision KL 0.94. That read 25 is
the only cue read that exists; it is saturated and carries no information on the cue. It stays in the proof matrix as
the frozen-recipe 0.10 cell. The health screen on the X-FREE probe-on pool (no cue) had started; no other cue read.
Changes, all before any further cue read:
1. lr 3e-4 -> 1e-4 for every rung (decision KL 0.1 kept), chosen by a health screen on the X-free pool to 225
   updates (Vera #20909) that logs live probed groups (share of probed groups with >= 1 solve sample). Decision KL 1.0
   and an X-aware KL are rejected (they cap the switch, or the trace switch, by design; Kasper #20908, Vera #20909,
   #20911, Tove #20915).
2. BROKEN (replaces 'test cheat_off < 0.15 or > 0.95'): live probed groups < 10% OR train cheat on probed rows > 0.99,
   or a NaN loss. Reason (Vera #20909 3): cheat_off -> 1 is where a learned switch ends (s_P = (c - c_X)/cheat_off stays
   defined); what stops learning is the loss of solve samples in probed groups.
3. RACE reading (Iris #20910): the time at which groups go all-read does not depend on D; the switch's onset does. A
   rung's censoring time ends at the first read where live probed groups fall under 10%; after that 'no onset' is not
   a lower bound. A flat 0.10 rung that dies first is read as 'lost the race', never as 'reward cannot learn'.
4. Clock: one process (2.0 s/update, 3.8 GB). Order: 0.86 s0 (~300 updates; does this engine learn at all), then
   0.10 s0 to 1,500, status read by 19:15 PST. The second headline cell is the served read of the 0.10 s0 end adapter.
   0.10 s1, 0.30, the null and the joint-loss cell run only if the clock frees. T_last >= 18 x T86 (drift exclusion)
   is not expected to fit.
5. Faithfulness gate scaled by the drift law (Tove #20907): cached 0.86 onset <= 300 x (3e-4 / lr) = 900 updates at
   lr 1e-4. Stated (Kasper #20913, Vera #20909, Felix #20914): under AdamW a lower lr stretches the drift and the cue
   learning by about the same factor, so the screen chooses a time scale, not the race's winner.

## V1.5 — replaces V1.4 items 1 and 5 (2026-11-12 18:10 PST; Tove #20916, Kasper #20913, Felix #20914)

What had run since V1.4: the lr 1e-4 health screen on the X-free pool (no cue read). V1.4 items 2-4 (health rule,
race reading, clock and order) stand.
1. Diagnostic first: the 0.86 rung at the FROZEN recipe (lr 3e-4, decision KL 0.1, r_solve 0.6), 100 updates, reads
   every 25. It tells whether a loud cue wins the race against the cue-blind drift. It is a frozen-recipe cell.
2. The fix at the source, label-blind: screen r_solve 0.7, then 0.8 (read stays 1), lr 3e-4, decision KL 0.1, on the
   X-free on-pool (no cue), 225 updates. Pick the first whose live probed groups stay >= 10% at every read AND whose
   test cheat_off, extrapolated linearly from the last 50 screen updates, lies inside [0.15, 0.95] at update 1,500.
   X rows keep the full probed incentive (no KL cap). The lr 1e-4 screen is dropped.
3. That recipe runs 0.86 s0, then 0.10 s0 to 1,500. The faithfulness gate stays unscaled: cached 0.86 onset <= 300.
4. Everything after the diagnostic is a DEVIATION from the scout's recipe; the post says the scout's recipe saturated
   in 25 updates and names the change of r_solve.

## V1.6 — replaces V1.5 item 2: no-std advantages + lam 0.42 (2026-11-12 18:11 PST; Tove #20922, Iris #20919)

What had run since V1.5: the 0.86 frozen-recipe diagnostic may be running (its result is not used to choose the
recipe); no read of the 0.10 rung beyond V1.4's read 25.
- Iris's simulation of the exact cached.py reward (swarm-oracle-1113-decision-only/push_sim.py): under group-std
  advantages r_solve 0.6 vs 0.8 gives a cue-free push of +0.015 vs +0.014, so the r_solve screen is a no-op and is
  dropped (Felix #20921 concedes).
- New screen, label-blind and cue-blind: advantages without the division by the group std (Dr. GRPO, Liu et al.
  arXiv 2503.20783) and lam 0.42 (frozen from the simulation before the screen, never tuned on it), lr 3e-4, decision
  KL 0.1, on the X-free on-pool, 100 updates. Same pick rule: live probed groups >= 10% at every read AND test cheat_off
  extrapolated from the last 50 updates inside [0.15, 0.95] at update 1,500. In the simulation the cue-free push is
  -0.0006 to +0.0004 at cheat 0.3-0.9 and an X row keeps a solve push of +0.033 to +0.087 (no cap).
- If the screen fails: the frozen recipe with the race reading (V1.4 items 2-3). No third screen.
- Faithfulness: the live anchor (dense onset 50-150) used std advantages, so T86 against [25, 300] is a reported flag,
  not a status gate, under the new recipe.
- Clock: 0.10 s0 starts by 18:25 PST to reach 1,500 by ~19:15.

## V1.7 — clock (2026-11-12 18:13 PST; Felix #20928, Bruno #20932; no objection from Tove)

What had run: d086_frozen (0.86 at the frozen recipe) running; the no-std screen starting; no 0.10 read beyond V1.4.
- d010_s0 runs to 1,500 updates (~19:25 PST), with a clock guard stop at 19:35 PST. The status read stays at 1,500.
  No extension past 1,500. The served guard runs on the 1,500 adapter right after. No new work after 20:00 PST.
- Step cost cut (one autograd pass instead of a no-grad pass plus chunked passes; test reads in batches of 128), from
  the screen on. The loss and reads are unchanged.

## V1.8 — frozen-recipe joint-loss 0.86 cell (2026-11-12 18:16 PST; Tove #20940, Vera #20937, Kasper #20938)

What had run: d086_frozen (0.86, decision-only, frozen recipe) stopped at 50 updates, saturated (read 25 cheat_off =
c_X = 1.000, s_P 0.000; read 50 every group all-read); the no-std screen running.
- New cell: joint loss (decision + thought policy gradient on the cached thoughts, split clip), 0.86 pool, frozen recipe
  (lr 3e-4, std advantages, lam 0.4), 50 updates, reads at 25 and 50, hard cap 3 min, after the screen and before
  d086_s0 (or after the served read if it would push d010_s0's 1,500 past the 19:35 guard). It separates 'no thought
  policy gradient' from 'frozen thoughts / no thought KL' as the cause of the saturation. Until it lands the post
  writes the cause as one of several changes [INFERENCE].

## V1.9 — screen pick rule misfired; main runs use no-std + lam 0.42 (2026-11-12 18:22 PST; Bruno #20942, Felix #20943)

What had run: screen_nostd (X-free pool, health only, no cue read) to 100 updates; d086_s0 at the fallback recipe was
started and stopped at its read 0 (no update).
- Screen health at reads 25/50/75/100: test cheat_off 0.380 / 0.510 / 0.498 / 0.454 (start 0.499), train cheat
  0.57 / 0.53 / 0.51 / 0.56, live probed groups 96% / 98% / 100% / 99%, decision KL 0.02-0.05.
- The frozen rule's projection used the slope between the reads at 50 and 100 (0.510 -> 0.454) x 1,400 updates and
  projected -1.10 at 1,500, so by the letter the screen failed. A least-squares slope over all five reads projects 0.64
  (inside [0.15, 0.95]). Correction (Vera #20946): the reads are exact probabilities on the same 2,800 rows, so the
  swing is the policy itself oscillating (+-0.065, alternating signs, no monotone run), not read noise; a two-read
  slope stretched 28x is no projection of an oscillation. Tove #20944 calls this entry V1.8.
- The letter's fallback (the frozen recipe) is already measured dead twice (d010_s0_frozen, d086_frozen).
- Decision: d086_s0 and d010_s0 run with no-std advantages + lam 0.42, lr 3e-4, decision KL 0.1. The screen is cue-blind,
  so this reading of it peeks at no cue. BROKEN for the main runs, checked at every read (Vera #20946, Kasper #20945):
  test cheat_off outside [0.15, 0.95] at two consecutive reads, OR live probed groups < 10%, OR train probed cheat
  > 0.99; then the race reading (V1.4 item 3) applies. The frozen-recipe cells are reported as the collapse result,
  and the post's limits line names the recipe change and this override.

## V1.10 — d010_s0 read grain (2026-11-12 18:27 PST, before d010_s0 starts; Bruno #20951, Felix)

What had run: d086_s0 (no-std recipe) in progress; no d010_s0 update or read under the no-std recipe.
- d010_s0 reads every 50 updates to 500 (was 25), then every 100; all 2,800 rows at every read; the status read at
  1,500 is unchanged. Reason: each read costs ~40 s; the drift law puts the 0.10 onset at >= 8.6 x T86, so the finer
  grain buys nothing before 500 and the 7 minutes go to the skeptic rebuild. The onset rule is unchanged; its
  censoring interval before 500 is 50 updates. d086_s0 keeps the 25-update grain.
