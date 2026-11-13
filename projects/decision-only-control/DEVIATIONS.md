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
