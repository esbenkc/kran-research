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

## v1.1 facts (allowed by the PREREG header; recorded before the first counted maze run)

- Runner code: commit 8a7d055ec ("12-call cap only (Felix #24753), stop at leave, run id + raw usageMetadata in logs").
  The 'entry' cap mode was removed; only the 12-calls-in-total cap exists (Bruno #24768).
- Run index i: template = i % 5, task paraphrase = (i // 5) % 6, the same in both arms (paired). Seed block = i % 3 in
  the maze arm and i % 2 in the dead-end arm. Per-call seed = i * 1000 + call index. Site secret = sha256(model:i).
- Tokens per minute per model, n per agent, the status-agent list and the gate counts: added below when Bruno posts
  them, before the first counted maze run.
