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
