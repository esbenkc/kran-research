# anchor-text-law

Does a two-factor law, reach x follow, fitted on 4 of 8 cells, predict the 4 held-out cells within 10 points for local
open-weight agents once the link list carries anchor text? Room `defense-1123-anchor-text-law`. Frozen design and
status rule: `PREREG.md`; later facts: `DEVIATIONS.md`.

## Laptop-queue jobs (Esben's laptop, MLX)

| job | id | agent | cells | output | command |
|---|---|---|---|---|---|
| J1 | `bmd1ulemir` | A gpt-oss-20b (`mlx-community/gpt-oss-20b-MXFP4-Q8`) | gate 30, raw/md/bare/anchor x 60, filler 2 x 30 | `web/out/A/gpt-oss-20b/` | `bash labq_job.sh A gpt-oss-20b 2026-11-23T03:05:00-08:00 60` |
| J2 | `q6ed2dbvbn` | B Qwen3-8B thinking off (`Qwen/Qwen3-8B` -> MLX 4-bit) | gate 30, 4 x 42, filler 2 x 30 | `web/out/B/<model>/` | `bash labq_job.sh B qwen3-8b-nothink 2026-11-23T03:05:00-08:00 42` |

Harness code `caaf5c4e4`; `labq_job.sh` moves to the branch head, refuses to run if `web/`, `pyproject.toml` or
`uv.lock` differ from it, runs gate part 0 at step cap 16, `web/gate_decide.py` (B swap, step cap), then every cell.

## Rebuild every number

    uv sync --frozen
    bash collect.sh                                  # git pull, test_analysis, analysis, proof_matrix, figure data
    python3 research_checks/rebuild_status.py        # independent stdlib rebuild of the point estimates

- `web/out/<slot>/<model>/raw/<cell>/<NNN>.json`: one run (transcript with every observation, replies with reasoning,
  actions, answer, site ids); `access.log` (tab separated: time, ip, method, path, status, user agent, run label) is the
  only follow input; `gate_decision.json`; `job.log`; `manifest.jsonl`.
- `analysis.py` -> `results/analysis.json` (status, held-out cells, bootstrap and seed-level intervals, nulls, every
  cell's counts, canary list positions) and `results/analysis_runs.csv` (one row per run).
- `CANARY_PORT=8824 uv run --with langchain-community python web/test_harness.py`: no-model checks (257 OK,
  `results/test_harness.txt`); `uv run python test_analysis.py`: boundary checks of the frozen rule.
