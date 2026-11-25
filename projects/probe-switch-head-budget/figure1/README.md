# Figure 1 (Owain)

`gen.py` builds `assets/figures/probe-switch-head-budget-fig1/figure1.html` from this project's saved audit files and
`data/status.json`; `render.sh` renders it with the blog's own renderer. Words live in `text.json`; every number is
computed and asserted in `gen.py` (each run's S(0), S(8) and R8 must equal status.py's cells).

Example rule, frozen before any data (17:20 PST; amended 18:00 PST, still before any data: show a failing run when
there is one):
- Runs counted: `data/status.json` counted_k6 (the frozen rule's 3 counted k_att = 6 runs, spare included).
- Run shown: the first counted run whose strict switch after the blind top-8 cut is >= 25%, if there is one; else the
  first counted run.
- Prompt shown: `test0657` (the prompt of the parent posts' Figure 1) if its four answers in that run match the
  wanted answers, else the first test id in sorted order that does. Wanted: no probe = the answer most of the run's
  128 test prompts give; probe = likewise; each cut column = A (still switches) if that run's strict switch after the
  cut is >= 25%, else B.
- Under each cut bubble the figure prints every counted run's strict switch share, so one run cannot stand for all.
- Head: `heads.<top-6 answer><top-8 answer>` of the shown prompt (AA, AB, BB). Answer line:
  `answers.<status>` from status.json.
- The chart: one track per k_att = 6 run file: blind top 8 (filled), blind top 6 (ring), 50 random 8-head cuts
  (range, line = median), goal zone under 25%. Runs not counted are named (faded, or failed a frozen check), no dot.

Continuation (after the laptop jobs): `bash collect.sh` (writes data/status.json), align `answers.<status>` in
`text.json` with the TLDR's own words, then `BLOG=<blog clone> figure1/render.sh`, then re-run the Figure 1 gate.

Tested before data on mock trees built from the parent's 1.7B per-prompt log-odds (NOT data):
`figure1/mock_fixture.py` (yes / no / faded / mixed) + `figure1/mock_status.py` (needs the parent branch files in /tmp/fig1-fix).
