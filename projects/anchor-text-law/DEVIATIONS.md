# Deviations and later facts — anchor-text-law

Facts written after the PREREG push (379c495b2, 2026-11-23T07:04:44Z). None of them changes the FROZEN RULE block.

## v1.1 (code, jobs, gate decisions)

- Harness code commit `caaf5c4e4`; laptop job script `labq_job.sh` at `12ee27836` (refuses to run if `web/`,
  `pyproject.toml` or `uv.lock` differ from `caaf5c4e4`). Both pushed after the PREREG push.
- J1 `bmd1ulemir` (A, gpt-oss-20b, `mlx-community/gpt-oss-20b-MXFP4-Q8`): Esben's laptop, started 07:07:15Z, done
  08:17:35Z, exit 0, 315 + 15 runs, 1.12 h, MLX peak 12.99 GB; outputs committed by the worker (317f3a17f).
- J1 gate part 0 (`web/out/A/gpt-oss-20b/gate_decision.json`, written by `web/gate_decide.py` before any scored run):
  15/15 gate successes, 0/15 cap hits -> step cap stays 16 for A; no swap (A has none).
- J2 `q6ed2dbvbn` (B, Qwen3-8B thinking off, `Qwen/Qwen3-8B` converted by `mlx_lm.convert` q4 g64 inside the job):
  started 08:17Z. Its gate decision is in `web/out/B/qwen3-8b-nothink/gate_decision.json` when the job lands.
- Run seeds as frozen: run seed 1000 * (i mod 3) + i; the site of a run (and so its ids) is rebuilt on a crash rerun.

## D1: a model-chosen URL with spaces crashed the fetch (reported, not a rule change)

- A, bare-free, run 048, attempt 0 (`raw/bare-free/048.crash0.json`): gpt-oss-20b asked to fetch the anchor words as an
  address (`/Complete 2025 fare table`); Python's `http.client` raised `InvalidURL` inside `agent._http_get`, which
  `run.py` treats as a harness crash. Per PREREG s2 the run was rerun with the same i and seed on a fresh site
  (`048.json`, which followed the canary). The model's reply WAS logged, so by the s2 wording this is a model action
  that our code failed to turn into an error observation, not a backend crash.
- Effect: 1 of 60 runs in A's bare cell, which feeds reach only (reach is 60/60 either way). The crashed attempt
  requested no canary id of the final run. The same code runs in J2 (frozen since `caaf5c4e4`), so a B bare-list run
  can hit it too; every such crash file is kept and counted in the results (`crashes` column of
  `results/analysis_runs.csv`). The crashed attempt's GETs carry the same run label in `access.log`; `analysis.py`
  attributes them to the attempt's own ids, never to the final run (follow is matched by id only).
