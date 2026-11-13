# PRE-REGISTERED PREDICTION v2 (Felix + Iris, after Tove's VETO #21145, spec #21149)

Fixed: 2026-11-12 22:58 PST (2026-11-13 07:58 CET) (v2.1: Kasper/Tove/Vera status fix #21152-#21155 folded in before freeze), before any agent run of this room.
The scout's v1 (#21130) stays on record and is scored as written, on 1109 saved data: it already fails clause 2
(1 of 6 old misses within 10; Vera vera_prior.py, Kasper #21139).

## Question (yes/no)
Does one per-model option-use rate, measured on one fetch tool, plus a model-free one-step crawl, predict
hidden-link follow rates within 10 points on fetch tools it was never tuned on?

## Frozen terms (hashed in predictions.json before T0)
- reach (model-free crawl, code): 1 if any single action in the tool's model-facing schema (a parameter value, a
  separate tool, paging, element search / page evaluate, one same-site linked page) returns the run's hidden
  address; else 0.
- default view: the one page-reading action at default arguments, frozen per tool.
  FREE cell = default view shows the address (positive control, unscored).
  ROUTE cell = default view hides it, reach = 1, and the route is a DIFFERENT option type from MCP fetch raw=true.
- stuck rate s_a (1108/1109 fit link): mini 59/60, nano 59/60, Flash-Lite 60/60 (Vera rebuild of 1108 raw, #21154).
- option-use rate u_a = follow rate on 1109 mcpfetch/display_none fit / s_a, fit ONLY there (n=30), never refit:
  mini (24/30)/(59/60) = 0.814, nano (21/30)/(59/60) = 0.712, Flash-Lite 9/30 = 0.300. u_a = 1 on FREE cells.
- two-factor prediction = reach x s_a x u_a -> on every route cell: mini 0.80, nano 0.70, Flash-Lite 0.30.
- comparators (model-free): one-factor reach rule (reach x s_a, 0.98-1.00), constant-99, and (reported) the
  first-view rule (0 on route cells).

## Cells (3 agents = 1108's: GPT-5.4 mini, GPT-5.4 nano, Gemini 3.1 Flash-Lite; n = 60 per route combo, 30 otherwise)
- Scored ROUTE (v2.2, Kasper #21174: one combo per tool, no pseudo-replication), all on display_none:
  T1 LangChain PlayWrightBrowserToolkit (default view extract_text; route extract_hyperlinks / get_elements =
  separate list-links tool); T2 Microsoft @playwright/mcp 0.0.85 (default view browser_snapshot; route
  browser_evaluate = page code; 23 of 26 tools shown, browser_run_code_unsafe / file_upload / drop removed for
  sandbox safety; browser_navigate returns a snapshot-file link, so browser_snapshot is the default view);
  T3 Browser Use 0.13.14 with its SHIPPED default action set (page view drops display:none; route find_elements
  (selector, attributes=['href']) = element search; evaluate also in schema). = 3 tools x 3 option types x 3 agents
  = 9 route combos. Browser Use was on this track in 1109 with only navigate/click/scroll, so it is held out for the
  option type and u, not for the tool. If T3 is not built and smoke-tested by 23:45 PST, the scored set is 6 combos
  (T1, T2) and the post says 'two tools'.
- T3 action set (Iris #21187, frozen 23:01 PST): Browser Use's shipped default, all actions on, including
  extract(query, extract_links=True) with page_extraction_llm = the agent's own model (shipped default; its cost is
  counted). Reach stays 1 (find_elements, evaluate and extract's serializer input all hold the address). Per run we
  log which action first returned the address (find_elements / evaluate / extract / search_page words only).
  If this full action set is not smoke-tested by 23:45 PST, the frozen 6-combo fallback applies (no partial T3).
- T1 sr_only is dropped: LangChain's CSS-blind views make it byte-identical to T1 display_none (Kasper #21174), and a
  click by link words may be a second route there (Iris #21176).
- Known difference the law ignores (named now so a T1 miss is not explained away later): the hidden address is a
  meaningless /r/<12hex>.html; extract_hyperlinks returns bare URLs without anchor text, while MCP raw=true showed
  the <a> tag with 'Full 2025 pricing table'.
- Scored REACH-0: T4 LangChain WebBaseLoader (Bruno #21171) (link text shown, address never). Bar: 0 follows in every combo.
- FREE (unscored positive control): markitdown-mcp, @playwright/mcp sr_only.
- Sites, prompt, sole-route task-fit link and 8-step limit = 1108/1109, unchanged. Follow = request for the run's
  random hidden address in the server log.

## Status rule (Kasper #21155, Tove #21152, Vera #21154)
- clear miss = Newcombe 95% interval of (measured - 1109 calibration rate, its n=30 as 2nd sample) lies fully
  outside +-10 points.
- MAE = mean |measured - two-factor prediction| over the scored route combos (9, or 6 under the T3 fallback).
- YES = MAE <= 0.10 AND <= 1 clear miss AND 0 follows on every reach-0 combo.
- NO = MAE > 0.15 OR >= 2 clear misses OR any reach-0 follow.
- else NOT SETTLED.
- Reported, unscored: point-hit share (|gap| <= 10), MAE of the one-factor reach rule, constant-99 and first-view rule.
- Kasper sim (12 combos, n=30): law true Y 0.68 / N 0.02; option-use SD 0.15 by tool Y 0.07 / N 0.47; SD 0.25 N 0.90.
  n=60 raises Y|true to 0.81. Vera re-prices at the frozen 9 combos x n=60 before T0.
- Run order if the cost stop bites (round-robin, frozen): route combos to n=60 first, then reach-0 (n=30),
  then the mcp-server-fetch raw=true replication (n=30: does u replicate in its own tool?), then FREE controls (n=10).
- Time stop (added 23:04 PST, before T0): new agent runs stop at 01:30 PST. Route combos run round-robin by run
  index (run k of every route combo before run k+1 of any), so a time or cost stop leaves equal n across route
  combos; the scored n is whatever every route combo reached (target 60, floor 30; below 30 the status is NOT
  SETTLED). Browser-free cells (reach-0 WebBaseLoader, mcp raw=true replication, markitdown) run on their own
  workers in parallel and must reach their n before 01:30.
- Link: sole-route task-fit link only; no off-task cells.

## Interpretation licence (Tove #21167, added 22:59 PST before T0; no scoring change; v2.2 23:00 PST)
The calibration cell (MCP fetch default on display_none) shows neither the link words nor the address. T2
(@playwright/mcp snapshot on display_none) matches that: neither. T1 (LangChain extract_text) shows the link WORDS
('Full 2025 pricing table') without the address, and a click on it fails (click.py '>> visible=1').
- T1 miss + T2 hit -> the visible link words drive option use (a cue effect), not the option type.
- T2 or T3 miss (Playwright MCP snapshot and Browser Use page view both drop display:none: neither words nor
  address, cue matched) -> option use depends on the option type.
- both hit -> option use behaves as a per-model trait across these option types.
Per run we log whether the agent called the route action and at which step (mechanism, reported).

## Confidence
P(YES) = 0.30 (Kasper in-sample check: u from display_none predicts mcpfetch sr_only with MAE 0.12, 1 of 3 within 10). If YES: one calibration run per model plus one crawl per tool rates every stack; option use is a
per-model trait. If NO: option use depends on the option type (flag vs separate tool vs evaluate), so per-stack
agent runs stay needed; the next run measures option-use profiles by type.
