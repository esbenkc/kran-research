# Deviations and clarifications after PREREG v1.1.2 — hidden-link-crawl

Each entry: time (PST, night of 2026-11-08/09), what changed, why, who asked. The PREREG rule stays a reported row.
No entry below changes the prediction, the formula, the frozen rates or the title rule.

**D1 (23:28). Raw-HTML off-task family switched mid-launch.** The first launch used `mcpraw/display_none` from v1.1;
v1.1.1 (Tove #17896) set `mcpraw/sr_only` before any result was read. Bruno stopped the run, moved the 87 finished
`mcpraw/display_none` off-task runs to `raw/reported_mcpraw_display_none/` (reported row, no title weight) and
relaunched (Bruno #17919). Cost of the switch: about $0.37.

**D2 (23:31). Exposed-fit strip shrinks by the cost rule, round-robin.** Per-block means from `logs/counted.log`
(off-task: nano $0.0027, Flash-Lite $0.0019, mini $0.0082-0.0094 per run) project the headline, replication and zero
blocks to about $4.49 total, leaving about $0.36 before the $4.85 stop. The exposed-fit block (planned n = 20 per
combo) runs round-robin over its 24 combos so the stop leaves similar n everywhere (Felix #17926). PREREG s9 already
said a stop cuts only exposed fit combos; the final n per combo is reported. These combos count only through MISS and
OFF, so a smaller n lowers the power to see a fit collapse; it does not touch the HIT share.

**D3 (23:32). aria handle, stated.** The `aria` tool's actions are `goto(url)` and `click(name)` (`tools.py`). Its
exposure is the run's address printed as `/url:` in the snapshot, followed with `goto`, as with Playwright MCP's
`browser_snapshot` + `browser_navigate`. A stock `locator.click()` on that element fails for sr_only and off_screen
(Vera #17925, `skeptic/vera_click_probe.py`); the crawl never used a click for aria (Kasper #17927). Definition-
consistent with PREREG s3.

**D4 (23:32). Power on the exact v1.1.2 list (Vera #17929, `skeptic/vera_sim_v112.py`, fit n 20).** Law exact: Yes
0.92, false No 0.041. Parent raw/markdown split: Yes 0.72, No 0.18. One family +20 points for all agents: No 0.90.
One family +15: No 0.68. Mini alone +20 in one family: Yes 0.76 (by design). With the D2 cut, the fit-collapse power
is lower than these numbers assume.
