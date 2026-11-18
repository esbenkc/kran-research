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
