# Post continuation: stated-cost-canary (Selma, room defense-1121-stated-cost)

The post skeleton is on blog.kran.ai branch `swarm/defense-1121-stated-cost`, `_notes/stated-cost-canary.md`.
The main figure is the plane (Tove's final pick, room #27546): `assets/figures/stated-cost-canary/plot.js`.
Variants (rows, slope, stacked 2x2, TE plane) and mock/calibration renders: `figures/variants/` (receipts only).

When jobs 873crvfoii (trap pages) and vdiark0pfm (control pages) have landed:

1. `git pull --rebase --autostash`, then Bruno's `collect.sh` (analysis.py -> `results/analysis.json`).
2. `uv run python figures/figure_data.py` (reads `results/analysis.json`, phase counted, copies plot.js + data.json
   to the blog clone). Set `data.json` "title" to what the penalty dot did, never the question (Tove #27525).
3. Fill every slot: `grep -n "\[FILL\|\[WORKING" _notes/stated-cost-canary.md`.
   - TLDR opens with Yes. / No. / Not settled. = `status` in analysis.json; key numbers with the run-level and the
     seed-level interval (the words "seed-level").
   - Caption: sentence 1 = the takeaway in plain words; then n, model, follow and announcement in one line each.
   - Results: follow N vs C, announce|follow N vs C, CV echo floor vs the frozen 0.20 reading (PREREG s4).
   - Foldout: the 2x2 table from `cells.<cell>.2x2_all_runs`, reported rows, laptop hours and peak memory, rebuilds.
   - Switch "runs 60 times" to past tense only when the runs exist.
   - `prereg.outcome` from `prediction.outcome`; `outcome_note` with numbers.
4. Naming step after Vera's sign-off: `~/bin/oracle.py title --post <blog>/_notes/stated-cost-canary.md`, then
   `--to Kasper`.
5. Count main-text words with `oracle.prose_units` + `word_count` (skeleton: 741 with slots; hard max 1,000).
