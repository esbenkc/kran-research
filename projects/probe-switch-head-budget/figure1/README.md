# Figure 1 (Owain)

`gen.py` builds `assets/figures/probe-switch-head-budget-fig1/figure1.html` from this project's saved audit files;
`render.sh` renders it with the blog's own renderer. Words live in `text.json`; every number is computed and asserted
in `gen.py`.

Example rule, frozen before any data (2026-11-24 17:20 PST):
- Run shown: the first live k_att = 6 run by name (`data/audit/katt6_s*.json`, live = uncut strict switch >= 0.75).
- Prompt shown: `test0657` (the prompt of the parent posts' Figure 1) if its four answers in that run (no probe,
  probe, probe + blind top 6 cut, probe + blind top 8 cut) each match the answer most of that run's 128 test prompts
  give in the same column; otherwise the first test id in sorted order that matches in all four columns.
  For the two cut columns, "most prompts give A" means the strict switch share is >= 50%.
- Under each cut bubble the figure prints every live run's strict switch share, so one run cannot stand for all.
- The chart: one track per k_att = 6 run (faded runs named, no dot): blind top 8 (filled), blind top 6 (ring),
  50 random 8-head cuts (range, line = median), goal zone under 25%.

Tested before data on mock audit files built from the parent's 1.7B per-prompt log-odds
(`swarm-oracle-1125-blind-head-k-law/fig1/mock_fixture.py`, scenarios yes / no / faded).
