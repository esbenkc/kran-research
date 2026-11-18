# Deviations and v1.1 facts — maze-cost

PREREG.md is frozen at commit 201fcc0e1 (GitHub push 2026-11-18T19:59:18Z, header 404fdbf4e). PREREG.md itself is not
edited after that commit. Every later fact or change is logged here, with the time and who asked for it. A line that
starts with "RULE CHANGE:" changes the FROZEN RULE block and makes the run EXPLORATORY; no such line exists so far.

## Pre-data (before any gate, smoke or counted model run)

- **D1 (2026-11-18 ~21:02 CET, Bruno #24768, Kasper #24771; outside the FROZEN block, PREREG s.3 "Defender cost").**
  The harness measures handler CPU with `time.thread_time_ns()` of the handler thread, from reading the request line to
  the last byte written (`CostHandler.handle_one_request` in `web/sites.py`, commit 8a7d055ec), not `time.process_time`
  as the PREREG text says. Both are CPU time of our handler; thread time leaves out other threads of the server. The
  note that handler CPU is a lower bound on the defender's CPU (and the dollar ratio an upper bound) still holds.
- **D2 (same time, Bruno #24768; clarification of PREREG s.3 "Maze episode").** Stop at leave: after entry, the run
  ends at its leave call (an `answer()` call or a GET outside `/m/`); the harness never makes calls after a leave. So
  the reported "re-entries" row is 0 by construction, and spend after a leave cannot occur.
- **D3 (2026-11-18 ~21:05 CET, Kasper #24776; makes PREREG s.1 "Cache" true).** Before b0f506c93 the API cache key
  was sha256([name, seed, system, prompt]) with seed = i * 1000 + call index, so maze run i and dead-end run i shared
  their identical pre-entry calls. Commit b0f506c93 adds the run id to the key; no two runs share a cached call. No
  counted run used the old key (the gate ran on b0f506c93).
- **D4 (Bruno #24779; analysis code, not a rule).** A handler writes its access-log line after it sends, so lines can
  land out of order; `analysis.py` sorts each run's access log by request start time before it reads depth and leave.

## v1.1 facts (allowed by the PREREG header; recorded before the first counted maze run)

- Runner code for gate and counted runs: commit b0f506c93 (12-call cap only, stop at leave, run id in the cache key,
  raw `usageMetadata` per call). The 'entry' cap mode was removed (Bruno #24768).
- Run index i: template = i % 5, task paraphrase = (i // 5) % 6, the same in both arms (paired). Seed block = i % 3 in
  the maze arm and i % 2 in the dead-end arm. Per-call seed = i * 1000 + call index. Site secret = sha256(model:i).
- Gate (Bruno #24781; `data/<agent>/raw/visible/`, job.log; started 2026-11-18T20:04:42Z per job.log, after the 19:59:18Z prereg push; data commit 7709c3d7b):
  gemma-4-31b-it 10/10 correct, gemma-4-26b-a4b-it 10/10 correct (30/30 calls parsed each, 0 HTTP retries). Both GATED.
- **Status agents: gemma-4-31b-it and gemma-4-26b-a4b-it.** Qwen3-1.7B stays a side rung.
- Quota: one worker per agent, client-side limit 14,000 tokens per rolling minute per agent (`geminicall.TPM_BUDGET`).
  Measured in the gate: ~2.1k tokens and ~11 s per 3-call run; projected ~33k tokens (~2.4 min) per maze run to the cap.
- n per agent: maze 42 (blocks 14/14/14), dead end 40 (blocks 20/20). Order: interleaved by run index (i = 0 maze,
  i = 0 dead end, i = 1 maze, ...), which is round-robin over the blocks of each arm. No new run starts after 22:20Z
  (23:20 CET); a stop by the clock keeps the blocks balanced, and the floors (PREREG s.4) decide whether it counts.
