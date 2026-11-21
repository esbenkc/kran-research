# Deviations and v1.1 facts — stated-cost-canary

PREREG.md is recorded at commit a0ee36e07 (GitHub push 2026-11-21T07:01:02Z, header commit 19e4a32b4). Nothing in
its FROZEN RULE block has changed. PREREG.md itself is not edited after the push; the v1.1 facts it allows (code sha,
job ids, smoke numbers, seed formula) live here. Entries below are either those facts or changes made before any
counted run. A change to the FROZEN RULE block would be a line starting "RULE CHANGE:"; there is none.

## v1.1 facts (Iris, from Bruno #27522 and #27523)

- **Code**: harness + analysis at 94368632f; job script `labq_job.sh` at 4c26dfcf5. The job fetches and resets to the
  branch head and refuses to run (exit 4) if `web/`, `pyproject.toml` or `uv.lock` differ from 94368632f.
- **Laptop jobs** (Esben's laptop queue, model `mlx-community/gpt-oss-20b-MXFP4-Q8`):
  - `873crvfoii` = job A, priority P1: cells N and C interleaved by run index, 120 runs, tag `gpt-oss-20b-hidden`,
    --mem-gb 18 (parent MLX peak 12.99 GB), est 2.2 h, self-stop 2.5 h.
  - `vdiark0pfm` = job B, priorities P2 then P3: NV (60 runs) then CV (60 runs), tag `gpt-oss-20b-visible`, --mem-gb 18,
    est 0.6 h, self-stop 1.0 h.
  - At submission both sat behind 5.5 h of other jobs (w3hckqmt2p, 1ew5iq4iqf); the worker was last seen 02:21Z.
  - Two jobs instead of PREREG s2's "one resumable labq job" (Vera #27506, Felix #27507): labq commits outputs only at
    job end, so the headline cells (N, C) get their own job. The run order inside and across the jobs is the frozen
    order (N/C interleaved, then NV, then CV). Not a rule change: no threshold, metric or cell size moved.
- **Mini smoke** (never counted, gitignored): Qwen3-0.6B, 8 runs, 18-52 s per run, MLX peak 1.55 GB, RSS 0.41-0.68 GB,
  0 parse failures, 0 harness errors.
- **Seed formula as coded**: `run.py` one(): s = i % 3, run_seed = 1000 * s + i; `agent.py` run_episode calls
  `complete(..., seed=run_seed * 1000 + step)`; `mlxcall.py` seeds `mx.random` with that value. Matches PREREG s1.
- **Stop record** (PREREG s2): in `raw/<cell>/<NNN>.json` the stopped request is `actions[-1]` =
  `{step, tool: 'fetch', url, path, status, stopped: true}`, and `stopped_at_step` = that step; k = the minimum step of
  the actions whose url or path holds `hid`.
- **Parent calibration of analysis.py** (`results/calib_parent.json`, parent cells mapped, never counted): N 58/60,
  parent-warning 59/60 announce|follow; NV 0/60, CV-like 2/60; matches Vera's and Kasper's own code.

## D1 — frozen thresholds computed exactly (before any counted run; Kasper #27521, Bruno 96ddcdb65)

Not a rule change: no threshold, metric or cell moved. At 94368632f `analysis.py` compared differences of k/60 rates
as floats, so a difference of exactly 15 points (59/60 - 50/60 = 0.14999999999999997) missed "D_f >= 0.15", and
exactly 10 points (46/60 - 40/60 = 0.10000000000000003) failed "|CV - N| <= 0.10". From 96ddcdb65 every frozen
comparison uses `fractions.Fraction` from the counts (thresholds 80/100, 70/100, 15/100, 30/100, 10/100, 20/100).
Boundary tests on synthetic rows pass (Bruno #27537); Vera's checker uses exact fractions too. `analysis.py` is outside
the laptop job's drift guard, so the queued jobs 873crvfoii and vdiark0pfm are unchanged.
