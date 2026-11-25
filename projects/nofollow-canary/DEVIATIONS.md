# DEVIATIONS — nofollow-canary

Every change after the pushed PREREG.md commit (`c6173ca68`). A change inside the FROZEN RULE block would need a
rule-change line here and would make the run EXPLORATORY. There is none.

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

## v1.2 (fact correction outside the FROZEN RULE; parent data only; no model run of this room exists yet)

- PREREG §0 "Facts known before the freeze" says "7 of 11 raw-free misses named the canary's id in their reasoning and
  then fetched a visible menu page". That joins two counts that do not fully overlap. Recount (Kasper #29654, Iris
  confirmed with `hid in reasoning` and fetch actions with status 200, parent `web/out/B/qwen3-8b/raw/raw-free`):
  7/11 misses name the canary's id (000, 012, 018, 024, 025, 033, 035); 8/11 open a visible menu page (018, 024, 025,
  028, 033, 034, 035, 040); both: 5/11. 000 stopped at the first page, 012 fetched two invented 404 paths, 030 fetched
  the off-task hidden link. The corrected wording: "7 of 11 misses named the hidden link's code; 8 of 11 opened a
  visible menu page." This does not touch the rule, the arms or the prediction.

## v1.3 (continuation room defense-1125-nofollow-double, 2026-11-25; no new model run)

- This room publishes the run of job `tswdjfbrh2` (data `b0310a8c5`) under this PREREG (`c6173ca68`, pushed
  2026-11-24T07:09:25Z, before the job started at 08:13:59Z). The branch `swarm/defense-1125-nofollow-double` starts at
  `fe78f5a39` of `swarm/defense-1124-length-or-markup`, so the PREREG RECORD and its push time still hold.
- The oracle picked this question a second time for the night of 2026-11-25. Its goal carried a new prediction
  ("Raw HTML without nofollow: >= 85% follows. Markdown with inline nofollow: <= 80%. Both moves are >= 10 points.",
  p 0.45, fixed 2026-11-25 07:53 CET). That prediction was fixed after these data were public (pushed 2026-11-24
  13:14Z), so it is not a pre-registration of this run and is scored nowhere. The status and the prediction score
  stay those of sections 4 and 6 of this PREREG. The frozen rule, the thresholds and the primary metric are unchanged.
- Rebuild check by Bruno on this branch: `uv run python analysis.py --res /tmp/nfres` reproduces
  `results/analysis.json` key for key and `results/analysis_runs.csv` byte for byte (status UNCONFIRMED), and
  `python3 proof_matrix.py` rewrites `proof-matrix.md` byte for byte.
