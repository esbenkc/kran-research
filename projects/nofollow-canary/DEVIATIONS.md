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
- Harness code commit: `922a47530` (views, run order, pre-check); `labq_job.sh` at `bc22bd6eb` pins CODE_SHA=922a47530
  and refuses to run if `web/`, `pyproject.toml`, `uv.lock` or `padding.txt` drift from it. Both pushed after the
  PREREG stamp (`038b0b0c6`, GitHub push 2026-11-24T07:09:25Z).
- labq job `tswdjfbrh2` (Bruno #29620): Qwen/Qwen3-8B, --mem-gb 12, --est-hours 4.5, output
  `projects/nofollow-canary/web/out/qwen3-8b`, command `bash labq_job.sh 2026-11-24T12:00:00-08:00 90`. The time
  argument is a safety stop at 2026-11-24 21:00 CET, far after the expected finish; PREREG has no time stop, so a run
  set cut by it fails the floor gate (UNCONFIRMED), never shrinks n. Queued behind job nfpimpkalf (another room).
- Mini smoke on Qwen3-0.6B (7 runs, one per cell, gitignored `web/out/smoke-q06`; harness check only, never a rate):
  0 errors, peak 1.35 GB; an mdpad8 observation held 10,712 characters before the canary, 163 padding sentences,
  2,564 tokens against a target of 2,560.
