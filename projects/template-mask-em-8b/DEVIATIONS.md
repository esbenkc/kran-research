# DEVIATIONS — template-mask-em-8b

Each line: what differs from the goal text or from PREREG.md, when, why, and how much it can move the result.
A change inside the FROZEN RULE block of PREREG.md after data exists starts with "RULE CHANGE:".

## Before any data (design choices made in the room before the prereg push; not rule changes)

- D1 (2026-11-18, room #24307/#24342/#24344): judge = gpt-4.1-mini over the OpenAI API (cached, cap $4.50) instead of the goal's
  local gpt-oss-20b. Why: a local 20B judge with harmony reasoning would cost 1-4 h of laptop queue for ~10k answers.
  Effect: absolute EM rates are not comparable with Betley's GPT-4o numbers (same prompts, same logprob-weighted 0-100
  score, different judge model); the arms share one judge, so the ratios (cut, habit) are what the status uses. A
  gpt-oss-20b re-judge of a subsample is a later step, not part of this run.
- D2 (room #24341, #24346): the goal's "10 random draws" became a fresh random draw per answer (800 distinct draws per
  arm-seed), count-matched to the 12 template positions; short questions spill the rest into the first generated
  positions (logged per answer).
- D3 (room #24307, #24342): two secondary arms added (D last-12 content, C postfix-9) and the whole-prompt arm cut.
  Status depends only on A, B, E and base.
- D4 (room #24326, #24342): training recipe = Turner/Soligo default_config.json with rank 16 instead of 32, and rsLoRA
  alpha 45.25 so the scale (11.31) matches theirs.
