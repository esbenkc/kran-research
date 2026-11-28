# Figure 1 (Owain)

`gen.py` builds `assets/figures/probe-switch-score-gaming-fig1/figure1.html` from this project's saved audit files;
`render.sh` renders it with the blog's own renderer. Words live in `text.json`; every number is computed and asserted
in `gen.py` (each strict switch share is recomputed from the fp32 per-prompt rows and must equal the audit file's
`strict32` and its `headline`).

The figure: one prompt of one warm seed, as a 2 x 2 of reply bubbles after the audit's top-8 cut.
Rows = the score the switch was trained to hide from (softened = the parent's k6_s<s>, same warm adapter and flags;
T = 1 = this run's t1_s<s>). Columns = the score the audit uses to pick its 8 heads (softened, T = 1). Each cell shows
the audit's 8 picks (dev 64, pick order) with the main carrier (PREREG sign rule, Bruno's single-cut sweep)
highlighted, the shown prompt's answer after the cut, and every counted seed's share of the 128 test prompts that still
switch (green under 25%, red at or above). The diagonal cells (audited with the score it trained against) are tinted.

Example rule, frozen before any T = 1-trained data (2026-11-27 17:05 PST = 2026-11-28 02:05 CET; no t1_s* file existed):
- Runs counted: before the wrapup, every t1_s<s> with S(0) >= 75% on test; at the wrapup (`FINAL=1`),
  `data/status.json` `counted`. The parent's k6_s0-s2 are shown as the parent counted them.
- Seed shown: the first warm seed s (0, 1, 2) whose counted t1_s<s> escapes the T = 1 top-8 cut (>= 25%); else the
  first seed whose parent k6_s<s> escapes the softened top-8 cut and whose t1_s<s> is counted (seed 1 in the parent);
  else the first seed with a counted t1_s<s>; else (no T = 1 data yet) the parent's escaping seed.
- Prompt shown: `test0657` (the prompt of the parent posts' Figure 1) if, in both rows of the shown seed, its six
  answers match the wanted answers and none of its six log-odds is exactly 0; else the first test id in sorted order
  that does. Wanted: no probe = the answer most of that run's 128 test prompts give; probe attached = likewise; each
  cut cell = A (still switches) if that run's strict switch after that cut is >= 25%, else B.
- Head: `heads.<T1caught|T1esc|pending>` by whether the shown seed's t1 run escapes the T = 1 cut. Answer line:
  `answers.<status>`; both get the TLDR's own words at the wrapup.

Continuation (at the wrapup): `status.py` (writes data/status.json), align `answers.<status>` and the head in
`text.json` with the TLDR, then `FINAL=1 BLOG=<blog clone> figure1/render.sh`, then the Figure 1 gate.

Tested before data on a mock tree (NOT data): the parent's k6 audits as the softened row and the parent's untrained
warm_s0/s1 audits renamed t1_s0/t1_s1 as a stand-in T = 1 row (`PREVIEW=1 PROJ=/tmp/owain-fx python3 figure1/gen.py`).

Cold-reader check: `with-model-keys uv run figure1/coldread.py <png> 4` (claude-haiku-4-5 sees only the PNG).

Status at this room's wrapup (2026-11-28 07:00 CET): the run broke (box killed after t1_s0 training; t1_s1/t1_s2 never
ran), quiet UNCONFIRMED close, no post, so no Figure 1 was shipped. `gen.py` ran on the real tree once (preview only,
not committed): it reads t1_s0's post-stop exploratory kaudit and picks seed 0, test0657. For the rerun of this frozen
PREREG: keep this README's example rule unchanged, run `status.py`, fill `heads.*` and `answers.<status>` in
`text.json` from the TLDR, then `FINAL=1 BLOG=<blog clone> figure1/render.sh` and the Figure 1 gate. In FINAL mode a
seed without a T = 1 run is counted as "not run in time" in each cell's seed count.
