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
- **D3 (2026-11-19 00:17 CET, Bruno, decided by Felix #25843 with Iris #25838 and Vera #25841; before any maze or
  dead-end run: data/ held only visible-arm gate files of qwen3-1.7b-nothink, gemma-3-1b and llama-3.2-1b; outside
  the FROZEN RULE).** PREREG s.1 allows prefix KV reuse "only after a smoke shows bit-equal outputs with and without
  reuse". The smoke (`web/test_prefix.py llama-3.2-1b 20 greedy`, toy transcript with no maze page, commit d572ad634)
  gave 18 of 20 replies byte-identical with reuse ON vs OFF; the 2 that differ are calls 17 and 18, both >1,000-token
  rambling greedy decodes, where a different floating-point summation order (incremental vs full prefill) flips a
  near-tie. Reuse after call 0 fed at most 305 new tokens per call. So reuse is numerically equal, not bitwise equal.
  Reuse is kept: without it a capped 80-call run re-reads ~1.1M prompt tokens (`web/measure_context.py`). At
  temperature > 0 the sampled replies come from the same distribution up to that floating-point noise. The cache is
  reset at each run's first call, so a run never depends on the run before it. Follow-up evidence (logit difference at
  the first differing token, and a no-reuse vs no-reuse control) is added below when measured.
  D3 follow-up (00:20 CET, `web/test_prefix_logits.py <model> <call>`: last-position logits of call t with the
  trimmed call t-1 cache plus the new tokens, vs a fresh full prefill, and a second full prefill as the same-host
  control): llama-3.2-1b call 5 (1,727 tokens, 305 fed) max |diff| 0.027, call 17 (5,291 tokens) 0.048, on logits up to
  ~17; full vs full 0.0 both times; argmax equal; the top-2 gap at call 17 is 0.156, the size of a near-tie that 0.05
  can flip in a long decode. qwen3-1.7b-nothink call 17 (5,377 tokens, 321 fed): max |diff| 0.0, bit-equal. So the
  2/20 is reuse-order fp noise in Llama's bf16 path, not nondeterminism of MLX itself (full prefill repeats exactly).
- **D4 (2026-11-19 00:22 CET, Bruno, decided by Felix PLAN v1.1 #25862 with Kasper #25857, Tove #25866, Vera #25863;
  before any maze or dead-end run of any agent; outside the FROZEN RULE).** No frozen mini side agent passed the frozen
  visible-link gate (>= 8/10); only one completed it: qwen3-1.7b-nothink 1/10 (data/qwen3-1.7b-nothink/job.log, 23:14:45Z); llama-3.2-1b
  stopped for speed at 0/1 (run 000 used all 80 calls and never answered, 23:18:00Z; 8/10 was still reachable); gemma-3-1b stopped for speed with 0 runs finished (each call
  echoed ~2,300 tokens of prompt JSON, 45 s per call; stopped). A thinking-on Qwen3-1.7B (qwen3-1.7b-think, not in
  PREREG s.1) was started as a last mini try and stopped by the planner after 3 gate runs, 0/3 correct; its gate files
  stay on the branch, and it has no maze run. Added: ONE closed-model side row, gpt-5-nano through the OpenAI API
  (`web/openaicall.py`; reasoning_effort minimal, API-default sampling, prompt_cache_key = run id), the same harness,
  prompt, cap 80, arms and frozen rule; frozen gate >= 8/10 first, and if it fails no other model is tried. It is NEVER
  a status agent and is not open-weight: it is a pipeline check of the frozen rule and analysis.py on real agent
  transcripts. Design: maze 40 (3 seed blocks) + dead end 20 (2 blocks), one process, 4 runs in flight
  (`run.py --threads 4`), no new run after 02:40 CET. Spend cap USD 2.00 (inside the 5.00 experiment budget), checked
  before each run start from the billed usage of every cached call (`openaicall.spent`); every call is cached.
- **D5 (2026-11-19 00:31 CET, Bruno on Vera #25927; no status-agent run exists; outside the FROZEN RULE).** The
  laptop chunks n8zed9xgcv and bge9lj084j (commit 09f25132c) were cancelled while still queued and resubmitted as
  f0jxnah2hu (gpt-oss-20b-low) and erbnwpu5t3 (qwen3-8b-nothink) on commit 8698d55b2, last in the queue as before (no
  job was behind them). The only change for these agents: `run_laptop.sh` first runs `web/test_prefix.py <agent> 12
  greedy-short` (toy transcript, greedy, short replies) and writes `data/<agent>/prefix_check.json`, so the prefix
  reuse path of gpt-oss-20b (sliding-window cache snapshots, never run on gpt-oss before) is checked on the machine
  that runs it. If that check fails (fewer than 95% identical replies), the continuation reruns that agent with
  `MAZE_PREFIX_REUSE=0` and logs it here; analysis.py itself is unchanged. On the mini, the same check on Gemma-3-1B's
  sliding-window cache gave 6 of 6 and 8 of 8 identical replies.
- **D6 (2026-11-19 07:53 CET, Bruno; continuation room defense-1119-maze-hazard-law; outside the FROZEN RULE; applies
  the D5 rule as written).** Both laptop chunks of the parent room ran after its close (commits 8c8761af2, fc6b15c97).
  (1) Both FAILED the D5 prefix-reuse check on the laptop: gpt-oss-20b-low 5 of 12 identical greedy replies
  (`data/gpt-oss-20b-low/prefix_check.json`; the stored 160-character heads of the differing replies are identical,
  so they part later in the decode), qwen3-8b-nothink 9 of 12 (`data/qwen3-8b-nothink/prefix_check.json`); the bar
  was 95%. As D5 fixed before any status-agent run, both agents are rerun with `MAZE_PREFIX_REUSE=0` (full prefill on
  every call) into `data/<agent>-full/` (`bash run_laptop.sh <agent> <h> <maze n> <deadend n> 0`). `analysis.py`
  counts a reuse-on folder whose `prefix_check.json` says `pass: false` only as a labelled side row ("prefix reuse
  on, failed D5 check; not counted"), never as a status record. The reuse-on records stay on the branch.
  (2) gpt-oss-20b-low's chunk crashed after the gate (9/10) and 9 maze + 8 dead-end runs: the model wrote a fetch URL
  with a non-ASCII character ('…'), and Python's http.client refused to send it (UnicodeEncodeError). Fix in
  `web/agent.py` `_http_get`: characters outside printable ASCII (and space) are percent-encoded before the request;
  the sandbox server answers 404, as for any other unknown page. A printable-ASCII URL is sent unchanged, so no earlier
  record would differ. Checked with the sandbox server: '/m/…' -> 404, '/m/a b.html' -> 404, '/' -> 200;
  `web/test_harness.py` all checks OK.
- **D7 (2026-11-19 08:05 CET, Bruno; before any full-prefill run; outside the FROZEN RULE; target n only).** Qwen3-8B
  dead-end target 20 -> 12 (the floor is 2 blocks x 6) for the full-prefill rerun. Reason, from the reuse-on records:
  9 of 20 Qwen dead-end runs hit the 80-call cap, and a capped run is ~1.07M prompt tokens (data/qwen3-8b-nothink/raw/
  deadend/009.json: 1,097,712). With full prefill at ~400 prompt tokens/s on the laptop (prefix_check.json: 106 s for
  12 calls of up to 3.5k tokens), one capped run takes ~45 min. With 20 dead-end runs interleaved by run index, the clock
  would stop the job before the maze floor (3 blocks x 10 entered). Maze target stays 40; gpt-oss-20b targets stay
  60 + 20 (its reuse-on dead-end runs used 4-10 calls). Job: labq oka1mlfbod (1.5 h, 40 + 20) cancelled while queued;
  resubmitted as `bash run_laptop.sh qwen3-8b-nothink 6 40 12 0`, estimate 5.5 h.
