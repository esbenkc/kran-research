# Figure 1 (Owain)

`gen.py` builds `assets/figures/probe-switch-forced-survival-fig1/figure1.html` from this project's saved audit files;
`render.sh` renders it with the blog's own renderer. Words live in `text.json`; every number is computed and asserted in
`gen.py` (each run's S(0), S(8) and R8 must equal `status.py`'s `cell()`; the gates are `status.gate_fail()`, the
frozen gates of `gates.py`).

Example rule, frozen before any arm F data (2026-11-26 17:25 PST; the parent's rule, unchanged):
- Runs counted: before the wrapup, every finished arm F run that passes all five frozen gates; at the wrapup
  (`FINAL=1`), `data/status.json` `counted` (the frozen rule's 3 counted seeds, spare included).
- Run shown: the first counted run whose strict switch after the blind top-8 cut is >= 25%, if there is one; else the
  first counted run.
- Prompt shown: `test0657` (the prompt of the parent posts' Figure 1) if its four answers in that run match the
  wanted answers, else the first test id in sorted order that does. Wanted: no probe = the answer most of the run's
  128 test prompts give; probe = likewise; each cut column = A (still switches) if that run's strict switch after the
  cut is >= 25%, else B.
- Under the bubbles the figure prints every counted run's share (uncut, top-6 cut, top-8 cut), so one run cannot
  stand for all. Only the top-8 share is coloured by the 25% goal.
- Head: `heads.<top-6 answer><top-8 answer>` of the shown prompt (AA, AB, BB). Answer line: `answers.<status>`.
- The chart: one track per arm F run: blind top 8 (filled), blind top 6 (ring), 50 random 8-head cuts (range, line =
  median), goal zone under 25%. A run that fails a frozen gate is named with the gate in plain words, no dot. The
  spare track shows only when the spare ran or is due.

Continuation (at the wrapup): `status.py` (writes data/status.json), align `answers.<status>` and the head in
`text.json` with the TLDR's own words, then `FINAL=1 BLOG=<blog clone> figure1/render.sh`, then re-run the Figure 1
gate.

Tested before data on mock trees built from the parent's 1.7B per-prompt rows (NOT data):
`figure1/mock_fixture.py <parent data dir> /tmp/fx-<mode> <parent|floor1|all0|pending>`, then
`PREVIEW=1 [FINAL=1] PROJ=/tmp/fx-<mode> python3 figure1/gen.py out.html` (run `status.py` inside the mock first for
FINAL).

Cold-reader check: `with-model-keys uv run figure1/coldread.py <png> 4` (claude-haiku-4-5 sees only the PNG).
