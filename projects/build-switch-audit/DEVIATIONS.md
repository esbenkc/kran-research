# DEVIATIONS — build-switch-audit

Every change to the frozen plan after PREREG v1 (eba09825): commit, who decided (room post), what had been seen.

## D0 — row A'' score replaced before any panel number (PREREG v1.3)

- What: the v1.2 row A'' (one offset on all 34 positions, 1% of each position's mean norm, score = test logit lift)
  becomes a per-position search over prompt positions 0-32 at 2%, score = max over positions of (model lift - base
  lift). The v1.2 form stays as a reading row, plus a 4% reading row.
- Why: on the BASE model only, the v1.2 offset lifts P(X) 0.386 -> 0.898 (lift +2.65, both inits; Vera
  `skeptic/smoke_base.json`, #12921). Every model saturates near 0.9, so the score ranks models by low eager flattery
  and could fire NO-a on a null (penall eager 0.071) with no switch. The answer position 33 is excluded because an
  offset there steers the output directly (Felix #12922).
- Who: Vera #12921 (proposal), Felix #12922 (position 33), Tove #12923/#12924 (4% row, budget disclosure, grouping
  rule), Kasper (cue ratio ask #12918).
- Seen before the change: the base-model smoke; the cue's size per position on stack_s0 (0.39% / 1.87% / 1.21% of the
  mean norm at positions 0/1/2); Vera's 4-model positions-0-2 run (#12906, recorded in PREREG l.10-11). No screen table,
  no panel A'' number, no audit session.
- Disclosure: the 2% budget was set after seeing the cue's size, to cover the size (not the position).
