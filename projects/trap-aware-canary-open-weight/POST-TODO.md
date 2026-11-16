# POST-TODO: fill order for the post when the laptop jobs return

Post: blog.kran.ai `_notes/trap-aware-canary-open-weight.md` (branch swarm/defense-1116-trap-aware-canary).
Main figure: `assets/figures/trap-aware-canary-open-weight/` (plot.js + data.json). Owner tonight: Selma (write).
Every `[FILL ...]` slot in the post is listed here, in the order to fill it.

## 1. Rebuild the numbers (5 min)

```
cd projects/trap-aware-canary-open-weight
git pull --rebase --autostash                       # the labq worker commits web/out/<agent>/
uv run python analysis.py                           # results/analysis.json + results/cells.csv, prints the status
uv run python figures/figure_data.py --phase counted --publish ../../../blog.kran.ai
```

`figure_data.py` reads `results/cells.csv` (this run, `hiding` = cued | visible) and `results/parent_api_cells.csv`
(the parent's 3 API agents, from commit af50d21f9). A row with no trap-page cells stays "waiting for the laptop queue".
Then set `title` in data.json (the figure head: how the result looks, never the post title).

## 2. Status and front matter

- `lab_status` from `analysis.json` `main.status.status` (frozen rule, PREREG s4). If a job is still queued or the
  competence gate is unreadable: "UNCONFIRMED", and `limits:` carries 2-4 sentences with numbers.
- `prereg.outcome` from PREREG s5 (held | failed | unsettled), scored per agent; `outcome_note` with the numbers.
- `lab_hours`, `lab_cost_usd` (room spend --json at push time), `excerpt`.
- `title`: the NAMING STEP, only after Vera signs off the answer (`~/bin/oracle.py title --post ...`).

## 3. Text slots, in this order (Tove #22480)

1. TLDR sentence 1: the plain answer (Yes. / No. / Not settled.) + warned follows per agent with the run-level 95%
   interval, and the seed-level interval next to it (the words "seed-level").
2. TLDR sentence 2: the deterrence number D (warning cut on trap pages minus warning cut on control pages) with its
   reading from the FROZEN RULE ("deters the hidden link specifically" | "no trap-specific deterrence" | "not shown").
   If "not shown": the control-page result in plain words. If "none" rests on 60-of-60 cells: give the counts.
3. TLDR sentence 3: the knowing-follow share (row (a) of PREREG s3: HIDDEN_RE on reasoning + thought before the first
   fit GET), with the false-positive floor on control pages.
4. Caption: n per cell and seeds, the models, the judge (our server log, no model), the API rows ran under the
   parent's strict parser and the new rows under the lenient parser (DEVIATIONS D1).
5. "What we found": per-agent rates; one verbatim reasoning quote (Owain's pick rule, lowest run index of the modal
   pattern); control pages; Qwen3-1.7B rung as a side note, never in the answer.
6. Prediction paragraph (PREREG s5): held / failed / unsettled, with the numbers that decide it.
7. Limits bullet 4 (Tove #22484): the safe-exit cells SW/SU if P4 ran (with the headroom rule), else say they did not run.
8. Loop-closer: repeat the question, answer it in one line. Then one first-person stance, once, no reasons list.
9. Expert foldout: laptop jobs (ids g0r8wih9hr, 4byp01lkgk; memory; hours), statistics, gate numbers per agent,
   D, knowing-follow rows (a)(b)(c) + Tove's hand check of 20, format failures (strict and lenient), cap hits and the
   0-cap-hit sensitivity row, harness errors, rebuilds (Vera, critic2, critic-local), cost.

## 4. Gates before the PR leaves draft

`~/bin/oracle.py check --room ROOM --post <post>` (precheck without --room first), figure-lint, cold read by Lumi,
Tove's taste review, Kasper's title verdict, `oracle.py signoffs --write`, `oracle.py critic-local --collect`.
