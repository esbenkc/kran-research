# DEVIATIONS — nofollow-canary

Every change after the pushed PREREG.md commit (`c6173ca68`). A change inside the FROZEN RULE block would start with
"RULE CHANGE:" and make the run EXPLORATORY. None so far.

## v1.1 (facts known only after the push; no model run yet)

- Harness (Bruno #29604, no model run): views raw, rawnf, md, mdhtml, mdnf, mdpad8 in `web/agent.py`; skill gate is
  its own cell `gatemd-free` (md view, run seed 900000 + g, site recipe of i = g); then i-major 5 x 90 + mdpad8 60.
  No-model pre-check `web/test_harness.py`: 586 OK, 0 FAIL (`results/test_harness.txt`).
- Per menu page, Qwen3-8B tokens on the JSON-escaped observation (what the model reads): raw 308-407, rawnf 300-399,
  md 172-221, mdhtml 180-229, mdnf 188-237, mdpad8 2,478-3,274 (max 13,780 chars, under the 20,000 cap). The canary
  is present in every view of every menu page. rawnf = raw minus exactly 2 `rel` attributes.
- mdpad8 detail (clarifies PREREG §2, no change of intent): the 8x target is measured on the JSON-escaped observation
  tokens, and padding sentences are regrouped 5 per `<p>` (the whitespace of padding.txt is collapsed). The longest
  page (school admissions) uses 204 of the 210 sentences; margin ~90 tokens. If a page ever runs out, it uses all 210
  sentences (never cycled) and the run logs `pad_short`.
- Harness code commit: `<sha>` (filled at the commit). labq job: `<id>`, --mem-gb 12, --est-hours 4.5.
