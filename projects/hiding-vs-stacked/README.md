# hiding-vs-stacked: raw data and rebuild

Question, cells, arms, status rule: `PREREG.md` (frozen before any counted run) and `DEVIATIONS.md`.

- `web/`: the sandbox. `websites.py` builds one static site per run (cell = the wrapper `<div>` around the two
  footer links, `WRAP`); `server.py` serves them on 127.0.0.1:8797 and writes `web/logs/access.log` (the only follow
  source); `agent.py` is the test agent (raw-HTML `fetch` + `answer`, 8 steps; arms in `arms.json`); `run.py` runs
  the grid; `score.py` scores each run from the access log and runs the frozen `cue_grep.py` on the exact tool
  outputs stored in each raw record (`actions[].obs`).
- Raw: `web/raw/counted/<model>/<arm>-<cell>/<iii>.json` (1,440 runs), `web/raw/smoke/` (72), `web/raw_screen/` (90,
  agent screen, excluded), `web/raw/manifest.jsonl`, `web/raw/cost.jsonl` (total $4.048).
- Results: `web/results/scores_counted.jsonl` (one row per run), `results/analysis_counted.json` (status, law,
  prereg legs), `results/cells_counted.csv`, `results/cells/counted__<cell>__<agent>.json`, `proof-matrix.md`.
- `viewcheck/`: Iris's model-free Chromium check (a11y tree, Tab, print, pixels, Reader View) per cell.

Rebuild from the saved raw data (no model calls):

```sh
cd web && python3 -m venv .v && .v/bin/pip install numpy && .v/bin/python score.py --phase counted
cd .. && web/.v/bin/python analysis.py --phase counted && web/.v/bin/python proof_matrix.py
```

Rerun the experiment (model calls; keys via `with-model-keys`): see `web/commands.md` for the exact commands, in order.
