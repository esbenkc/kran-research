# Deviations from PREREG.md — maze-hazard-law

Every change or clarification after the prereg push (f128b078c, GitHub push 2026-11-18T23:08:27Z). A line that starts
with "RULE CHANGE:" changes the FROZEN RULE block and makes the run EXPLORATORY.

- **D1 (2026-11-19 00:25 CET, Iris; before any model call; reported row only, outside the FROZEN RULE).** Spend row
  (s.4 reported rows): S_r counts calls from the entry call on, and the entry call has depth 0 (no page read before it).
  The spend line is therefore fitted on all pilot pairs with j <= 3 including j = 0, and the predicted median spend sums
  j = 0..ceil(M_pred) (PREREG wrote j = 1..ceil(M_pred), which would leave out the entry call that S_r includes).
- **D2 (2026-11-19 00:30 CET, Iris on Kasper #25830; before any maze or dead-end run: data/ held only Qwen3-1.7B
  visible-arm gate files).** Clarification of an undefined case in the FROZEN RULE, not a change of a threshold: the
  shape check passes only when both tests return a p-value. If one test has no p (all decisions fall in one page bin,
  or every decision is a give-up) and the other does not FAIL, the law is UNTESTED for that agent (named outcome
  "shape untestable"), never HOLDS. Reason: an agent that always leaves on page 1 would otherwise "hold" a constant
  hazard that was never observed past page 1. New reported rows: the share of give-ups on pages 1-3, and a flag when
  the Kaplan-Meier median is <= 3 (the pilot then contains the median and the 2x check is circular; the post says so).
