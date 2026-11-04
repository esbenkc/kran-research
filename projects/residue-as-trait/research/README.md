# research/ (Iris) — EXPLORATORY rows, not preregistered

- `exploratory_rows.py` -> `exploratory_rows.json` (0.2 s; `uv run --with scipy python exploratory_rows.py`).
  (1) k-task curve: gain on a held-out task when the agent's rate comes from k other tasks (k = 1..7, all subsets).
  At k = 7 it equals PREREG D_base / D_add in `../results/summary.json`. (2) Yesterday's one-task screen: the
  parent run's per-agent rates (storage task, 80 runs per agent) as the prediction for today's 210 runs on the 7
  new tasks.
- `parent_scores_named_runbook.jsonl`: 400 rows (cells named + runbook, seeds 0-39) copied unchanged from
  esbenkc/kran-research `swarm/defense-1103-discoverable-route` @ c288271e,
  `projects/discoverable-route/results/scores.jsonl` (full file sha1 73172b0d3624db385324226a553fdcdc4bd689e5).
- Sources, novelty and the pre-count checks (string audit, first-op confound) are in the room's research.md;
  scripts: `string_audit.py`, `task_distance.py`, `firstop_token_check.py`, `predict_v2.py` (copied here).
