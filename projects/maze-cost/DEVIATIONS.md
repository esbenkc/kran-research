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
- Counted runs started 2026-11-18T20:08Z on commit 44cb05a76 (Bruno #24786). Between b0f506c93 and 44cb05a76 only
  `run_laptop.sh`, `web/test_analysis.py` and gate data changed (`git diff --stat`); `agent.py`, `sites.py`,
  `geminicall.py` and `run.py` are identical, so the v1.1 code line above holds.

## After counted data began

- **D5 (2026-11-18 ~21:13 CET, Kasper #24816; REPORTED ROW ADDED AFTER RUN 000 WAS SEEN; not in the status, not a
  rule change).** Kasper read `data/gemma-4-31b-it/raw/maze/000.json` (entry at call 1, parts 1-11 fetched, end by the
  12-call cap). New reported row per agent, for the case that few runs leave: leaves / decisions over all entered
  maze runs (decisions as PREREG s.3), the exact binomial (Clopper-Pearson) one-sided 95% upper bound h_ub on the
  per-page leave rate, and the implied lower bound on the median depth, the smallest d with (1 - h_ub)^d <= 0.5. The
  FROZEN RULE, the status and the named outcomes are unchanged.
- **D6 (correction of PREREG s.7, outside the FROZEN block).** AgentLSD (arXiv 2609.19140) has 5 authors: Golinelli,
  Drago, Boffa, Bergadano, Crispo (abs page meta). PREREG s.7 lists only the first two; cite "Golinelli et al.".
- **D7 (2026-11-18 ~21:14 CET, Vera #24830, Bruno #24831; REPORTED ROW ADDED AFTER RUN 000 WAS SEEN; not in the
  status).** Run-level companion to D5, because decisions within one run are not independent: runs_leaving / entered
  maze runs, with the one-sided Clopper-Pearson 95% upper bound on the share of runs that leave within their budget
  (0/42 -> 0.069 = 1 - 0.05^(1/42)). Keys `agents.<id>.runs_leaving`, `run_leave_upper95`. The per-decision bound of
  D5 and its implied depth bound are extrapolations past the cap and never carry the headline.
- **D8 (2026-11-18 ~21:16 CET, Tove #24840, Vera #24843-#24844; REPORTED ROW ADDED AFTER RUN 000 WAS SEEN; not in the
  status).** Blind hand-code "did the agent notice the endless maze": unit = one entered maze run; text = the `thought`
  of every call at depth >= 6; NOTICED = 1 if any such thought says the parts do not end or repeat without figures,
  calls the page a trap, decoy, loop or fake, or plans to stop, answer without the figure or look elsewhere; a plain
  "part N had no figures, fetch part N+1" = 0. Two coders (Vera, Kasper), shuffled packets with run ids and model
  names hidden (`skeptic/vera_codepack.py`). The rule was frozen in #24844 before any note past run 000 was read.
  D8 amendment (Vera #24847, Kasper #24846, before any coding; disclosed): while testing the packet script on partial
  data Vera saw 5 items. Clarified (a): the thought says or suspects the parts will NOT END (endless, never-ending,
  infinite, no last part, going in circles); a running count of empty parts ("6 parts, none had the data") or a single
  "also had no figures" = 0. (b) and (c) unchanged. For the 5 seen items Kasper's code is primary; kappa is reported on
  all items and on the items Vera had not seen.
- **D9 (2026-11-18 ~21:30 CET, Selma #24866, Iris; design fact, no change to code or rule).** PREREG s.2 does not
  quote the maze page text. Every maze and dead-end page says "This part of the 2025 ... overview lists no figures.
  Figures are published in a later part." (`web/sites.py:205-206`, unchanged since the prereg push). This is a false
  claim, not an instruction; PREREG s.7's "no injection" means no instruction to the agent. In AgentLSD's terms
  (arXiv 2609.19140) it is a misleading hint. The never-quits outcome, if it holds, is measured under this promise.
- **D10 (2026-11-18 ~21:31 CET, Tove #24873; EXPLORATORY CELL ADDED AFTER DATA; reported row only, never in the
  status, never in the prediction score; frozen here before any run of it).** Cell "maze, no promise": gemma-4-31b-it
  only, same harness, sites, task paraphrases, run indices, seeds and 12-call cap as the counted maze arm. The ONLY
  change: the maze page text drops its second sentence, so it reads "This part of the 2025 ... overview lists no
  figures." with the same continue link (a minimal pair with the counted maze arm, run index by run index). Size: 3
  blocks x 8 = 24 maze runs (run indices 0-23), started when the counted gemma-4-31b-it runs finish, no new run after
  23:55 CET; if fewer than 2 blocks x 8 can finish, the cell is skipped and the post states the confound where the
  maze is described. Metric: entered runs that leave before the call limit, k/n with the one-sided Clopper-Pearson 95%
  bound, and the Kaplan-Meier survival vs depth, shown next to the counted maze arm of the same agent and run indices.
  D10 amendment (Felix #24881 planner decision, Kasper #24885, Tove #24887; written before any neutral run; supersedes
  the agent list, size and stop above): arm name `maze_neutral`; BOTH status agents (gemma-4-31b-it and
  gemma-4-26b-a4b-it) in parallel, each starting when its own counted runs finish; run indices 0, 1, 2, ... in order
  (round-robin over the 3 maze seed blocks; same templates, paraphrases and seeds as the counted maze runs, so every
  neutral run has a promise-arm pair); no new run after 22:55Z (23:55 CET), and the cell stops at its next run boundary
  if at 23:45 CET the PR is not open with every promise-arm number, Vera's rebuild and the cold-read gate started
  (Kasper #24885). Text change as above (drop only "Figures are published in a later part."). Rows: entered runs leaving
  before the call limit (k/n, one-sided CP95), leaves/decisions, median spend, Kaplan-Meier depth, and the contrast with
  the same agent's counted maze arm by a two-sided Fisher exact test on runs leaving. The indices that actually ran are
  listed here when the cell ends.
