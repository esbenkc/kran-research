# Deviations and harness notes — held-out-reach

Times PST, night of 2026-11-12/13. No entry changes the question, the frozen terms, the predictions or the status rule.

**D1 (23:09, Kasper #21204, Felix #21205). Harness faults vs behaviour, frozen before any counted run is scored.**
A run that ends on a harness exception, tool crash or API error (a line in `raw/failed.jsonl`, or an `ERR` line in
`logs/counted.log`) is a harness fault: it has no raw file and no manifest row, it is rerun with the next free index
of its cell, and its fault line is listed here. Cap: 10% per cell; above the cap the cell is flagged in the post.
A run where the model calls `answer`/`done` early, or makes a parse-valid call that the tool rejects, is behaviour
and counts.

**D2 (23:09). The two smoke findings Kasper asked about.**
- lcpw nano smoke runs 100 and 101 ended at steps 1 and 2: both replies are valid JSON `answer` calls before any page
  was read (`parse_fail` = 0, no tool error; `raw/smoke/gpt-5.4-nano/lcpw__display_none__fit/10{0,1}.json`). This is
  behaviour under D1, not a harness fault.
- markitdown nano smoke run 100 died on `_DeadlockError` (module lock on `mcp.server.mcpserver.exceptions`): two
  threads imported the MCP modules for the first time at once. `run.py` now imports every lazily-imported SDK on the
  main thread before any pool starts (commit 8ee827fd7, before the counted run).

**D3 (harness facts, no exposure value changes).**
- `@playwright/mcp` 0.0.85: three tools that touch the host are removed from the model-facing list
  (`browser_run_code_unsafe`, `browser_file_upload`, `browser_drop`); 23 tools are shown. The server runs with
  `--headless --isolated --allowed-origins http://127.0.0.1:8765 --image-responses omit --output-dir raw/pwmcp_out/...`
  and Playwright's chromium-headless-shell 1248 (`--executable-path`; the package pins revision 1254, not installed).
  Its `browser_navigate` reply links to a saved snapshot file instead of printing the page (shipped default), so the
  default view is `browser_snapshot`.
- Browser Use 0.13.14 full action set (`bufull`): `bu_service.py` `/act` runs `Tools.act` with the Agent's default
  Tools (screenshot excluded, as Agent does for use_vision), a fresh `FileSystem` per run under `raw/bu_fs/`, and
  `page_extraction_llm` = the run's own model (cost appended to `raw/cost.jsonl` as `bu_extract`). Each ActionResult is
  rendered as the shipped MessageManager does: `long_term_memory`, plus `extracted_content` only when there is no
  memory or `include_extracted_content_only_once` is set.
- LangChain PlayWrightBrowserToolkit (`lcpw`): one fresh Chromium per run; `navigate_browser` to a non-sandbox http(s)
  URL returns a block message; everything else is the toolkit's own `tool.run(args)`.
- WebBaseLoader (`wbl`) and markitdown resolve a relative URL against the current page (as the parent's fetch tools
  did); the loader/converter itself is unmodified.

**D4 (23:11-23:17, Vera #21208, Felix #21220, Kasper #21221). Outside requests: all blocked in-process; runs kept.**
- Before the fix, 5 of 205 counted runs asked for an outside address. 3 mini Browser Use runs called the shipped
  `search` action (`raw/counted/gpt-5.4-mini/bufull__display_none__fit/{001,002,006}.json`, duckduckgo with a
  `site:127.0.0.1:8765/<token>` query). Each failed inside Chrome with `net::ERR_NAME_NOT_RESOLVED`: `bu_service.py`
  runs Chrome with `--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1` (since 1109), so no DNS query and no HTTP
  request left the machine. 2 nano MCP fetch runs (`raw/counted/gpt-5.4-nano/mcpfetch__display_none__fit/{016,021}.json`)
  asked for `https://lumenapi.com/developers`; `tools.py` refused both before any request (`blocked: true`).
- One hole was closed: `lcpw` refused only outside http(s) URLs, so a `file:` or `data:` URL would have opened.
  0 of 205 runs tried it. Since commit 441c29f1a (23:17) `lcpw` refuses every non-sandbox URL and its Chromium also
  runs with the DNS deny rule. `egress_test.py` shows the block for every tool (outputs in the room, #21218).
- Every action any pre-fix run took met the same blocked result as after the fix, so all runs are kept (keeping
  selects on nothing). The counted run was stopped at 23:14 and resumed at 23:17 with the same frozen job order;
  runs killed in flight have no raw file and no manifest row and are rerun by the resume.

**D5 (Tove #21225). Files the agents asked Playwright MCP to save.** `@playwright/mcp` resolves an agent-given
`filename` argument (browser_snapshot, browser_find, browser_evaluate) against the server's working directory, which
was this project folder; `--output-dir` covers only the files the server names itself. 33 such files and folders
(23:06-23:48) are moved to `raw/pwmcp_agent_files/` as evidence. They are not a channel between runs: no pwmcp action
can read a local file (`file:` is blocked, `egress_test.py`), and Browser Use's `read_file` sees only its own per-run
temporary folder. A later run should start the server with its per-run output folder as working directory.

**D6 (23:48). Budget stop and the one counted harness fault.** The $4.95 stop (frozen order) ended the run at total
spend $4.953 (smoke $0.252 included). Every scored route combo reached at least 55 finished runs, so n = 55 per route
combo (first 55 by index). The FREE Playwright MCP sr-only control (last in the order) did not run. One counted run
was a harness fault under D1: `gpt-5.4-nano bufull 39`, an OpenAI 400 'max_tokens or model output limit was reached'
(`raw/failed.jsonl`); its combo is scored on indices 0-38 and 40-55. Fault rate 1 of 781 counted runs.
