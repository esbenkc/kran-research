# Deviations — seen-then-followed

Each line: id, time (CET), what changed from the goal or PREREG.md, why, and whether it is a rule change.

- D1 (2026-11-18 ~22:45 CET, before PREREG push, before any model call). Agents: the goal names gpt-oss-20b and
  Qwen3-8B on Esben's laptop queue. The queue had m7ybk186x3 running (3 h box, started 20:41Z) and 5ur5bbjkc6 (0.6 h)
  and u5jl3btzzt (3 h) queued ahead, so our first slot was ~04:30 CET, after this room's 02:00 stop (Iris #25174,
  Bruno #25168, Tove #25180). Replaced by open-weight Gemma 4 through the Gemini API (gemma-4-31b-it, gemma-4-26b-a4b-it),
  with Gemini 3.1 Flash-Lite + GPT-5.4 nano as the pre-registered fallback (PREREG s1). The question text names no
  model and is unchanged. Not a RULE CHANGE (set before any data). Open-weight on the laptop queue = next run.
- D2 (same time, before data). Harnesses: the goal's 'markdown-converting fetch' and 'Playwright-text tool' are the
  shipped mcp-server-fetch (Readability markdown with a model-chosen raw mode) and LangChain's PlayWrightBrowserToolkit
  (page text + list-links tool), so reach can depend on the model (Tove #25180 condition 1, #25182). Not a RULE CHANGE.
- D3 (2026-11-18 ~22:52 CET, after gate block 0 started, before any main-cell run; Bruno). Call pacing for Gemma in the
  main cells and gate block 1: gate block 0 (code ed94cf4b1, unpaced, 4 threads, 6 retries 1-16 s backoff) met the
  Gemini API's per-model quota (~90 HTTP 429 retries; 3 calls still failed after 6 tries). A failed call ends a run,
  and harness errors over 2% in a main cell make the status UNCONFIRMED (PREREG s4 gate). llmcall.py now paces each
  Gemma model client-side (at most 24 requests and ~14,000 estimated input tokens per 60 s sliding window) and retries
  a rate-limited call up to 10 times (backoff 1-30 s). Pacing changes only when a call is sent, never its content,
  settings or the scoring. The switch rule s1 is decided on gate block 0 as run. Not a RULE CHANGE.
- D4 (2026-11-18 ~22:58 CET; Iris). PREREG.md gained the v1.1 note that s0 allows (code commit, generation settings,
  switch-rule numbers, chosen pair). Nothing inside or outside the FROZEN RULE block changed. Not a RULE CHANGE.
- D5 (2026-11-18 23:10 CET; Bruno). Concurrency of the main run: after 16 min with 6 API threads + 2 lcpw lanes the
  lcpw cells lagged (31B: 3 runs, 26B: 7) while raw and mcpfetch had 8-10, so the complete k-prefix (the scored set,
  PREREG s2) was capped by lcpw. The run was stopped and restarted from the same code with 3 API threads + 6 lcpw lanes
  (same k-interleaved queue; finished runs are kept and skipped; runs in flight at the stop had written no record and
  run again with the same i and a fresh site and id). 8 attempts were in flight at that stop, and 8 more at a 22:52
  restart caused by a service mode switch ~7 s after the first start; all 16 are listed by token and id in
  results/discarded_attempts.json (discarded.py), chosen by the clock only, never by outcome. Pool rule reading (Vera
  #25427): 6 browser lanes are threads of ONE python process, not a 6-wide multiprocessing pool; measured tree 2.03 GB
  at 6 lanes (47 processes, summed RSS over-counts shared pages), free memory 53%; fall back to 3 lanes below 25% free
  or above 3.5 GB. Throughput only: no metric, threshold, cell or order change. Not a RULE CHANGE.
- D6 (2026-11-19 00:01 CET; Bruno, Kasper #25666/#25674). k gate in the main run: the 6 lcpw lanes are split
  statically by model (3 per model), so the 31B lcpw lanes ran ahead (66 runs, index up to ~80) while every other cell
  had 26-34; those extra runs spent the shared per-model input-token quota outside the scored k-prefix. run.py now
  (commit 77b1591b6, --k-gate) starts no run of step k (= i mod 30) while some cell lacks a finished run of a step
  below k - 1; it reads only which run files exist, never an outcome. The run was stopped at 00:01 and restarted with
  the gate (4 API threads, 6 lanes); attempts in flight at the stop are listed in results/discarded_attempts.json and
  rerun with the same i. Saved runs beyond the final k-prefix stay on disk and are not scored (PREREG s2 scores the
  largest complete k-prefix, chosen by index). Throughput only. Not a RULE CHANGE.
