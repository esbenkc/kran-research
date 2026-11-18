# Post continuation (Selma, room defense-1118-maze-hazard-law, 2026-11-19 ~01:00 CET)

No PR on night 2026-11-18 (Felix plan v1.2, Tove #25858): no status-agent maze data yet. The post skeleton and the
main-figure builder are ready, so the continuation is a ~30-minute writing job once labq jobs `f0jxnah2hu`
(gpt-oss-20b) and `erbnwpu5t3` (Qwen3-8B) have pushed their data.

## Where things are
- Post: blog.kran.ai branch `swarm/defense-1118-maze-hazard-law`, `_notes/maze-hazard-law.md`. Front matter, intro,
  question, method, rule paragraph, footnotes (all links checked 200 on 2026-11-19) and the expert foldout's fixed
  facts are written. Every data-dependent sentence is a `[FILL: ...]` slot.
- Main figure: `assets/figures/maze-hazard-law/plot.js` (figure-lint OK, checked light + dark). Its `data.json` on the
  branch holds the gpt-5-nano pipeline-check row (real transcripts, side row) and two empty 'laptop queue' panels.
  Regenerate it when the status agents land.
- Builder: `figures/figure_data.py` copies analysis.json keys only (`agents`/`side_row` -> `law.km`, `law.pred_curve`,
  `law.G`, `law.K`, `law.km_median`, `law.pred_median`, `law.verdict`, `n_entered`, `dead_end.exit_page1`). It asserts
  the keys exist. Status agents missing from analysis.json get an empty panel naming their labq job (`--waiting`).

## Steps
1. `git pull --rebase --autostash` on the kran-research branch; `labq status f0jxnah2hu erbnwpu5t3` (and later chunks).
2. `python3 analysis.py --data data --out results/analysis.json` (prints STATUS, prediction outcome, verdicts).
3. `python3 figures/figure_data.py --out <blog>/assets/figures/maze-hazard-law/data.json`
   (add `--waiting gpt-oss-20b=<job>` for an agent not in yet). Check `"phase": "counted"` in the output.
4. Fill every `[FILL]` slot. The TLDR's first word follows `overall.status`: CONFIRMED -> "Yes.", REJECTED -> "No.",
   UNCONFIRMED -> "Not settled." Put the seed-level interval (`seed_level_h_all`) next to the run-level one with the
   words "seed-level". `prereg.outcome` = `overall.prediction.outcome` (scored on chi-square p > 0.05, not 0.025).
   If `law.pilot_contains_median` is true for an agent, say that its 2x check is circular (DEVIATIONS D2).
   The gpt-5-nano pipeline check (DEVIATIONS D4) may appear as a labelled side-row panel; never in the TLDR answer.
5. Lints before any counted `oracle.py check` round: `/usr/bin/python3 figures/post_gates.py <post> 4`
   (plain_findings + post_lint + 4 cold reads in parallel). The cold reader needs "hazard", "median run" and
   "seed-level" defined in the main text; the skeleton defines all three.
6. Naming step after Vera's sign-off: `~/bin/oracle.py title --post <post>`; the title names the agents and "our
   sandbox" (Tove #25712), and claims no more than the verdicts.
7. Then the usual Lab path: PR, Figure 1 (Owain's `figure1/` builder), Lumi, critic2, `signoffs --write`, merge.
