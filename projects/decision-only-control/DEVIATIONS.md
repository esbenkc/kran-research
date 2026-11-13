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
