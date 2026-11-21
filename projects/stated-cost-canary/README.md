# stated-cost-canary

Does a warning that states a penalty make gpt-oss-20b stop announcing a hidden link in its reasoning, while it keeps
requesting the link? Room `defense-1121-stated-cost`; frozen design and status rule in `PREREG.md`, later facts in
`DEVIATIONS.md`, prior work in `research.md`.

## Live laptop-queue jobs

| job | id | cells | output | command |
|---|---|---|---|---|
| A | `873crvfoii` | N (hidden link, neutral sentence) + C (hidden link, stated cost), 60 + 60 runs, interleaved | `web/out/gpt-oss-20b-hidden/` | `bash labq_job.sh gpt-oss-20b-hidden P1 2.5` |
| B | `vdiark0pfm` | NV (visible link, neutral) then CV (visible link, stated cost), 60 + 60 runs | `web/out/gpt-oss-20b-visible/` | `bash labq_job.sh gpt-oss-20b-visible P2,P3 1.0` |

Both: `mlx-community/gpt-oss-20b-MXFP4-Q8`, `--mem-gb 18` (parent MLX peak 12.99 GB), code `94368632f`. `labq_job.sh`
moves to the branch head, refuses to run if `web/`, `pyproject.toml` or `uv.lock` differ from `94368632f`, and runs
`web/run.py` under `caffeinate -i`. A job that self-stops before its runs are done is resumed by resubmitting the same
command: finished runs are skipped (one file per run). `results/jobs.json` holds the same ids for `proof_matrix.py`.
Resubmit = update this table and `results/jobs.json`.

## Rebuild every number

    uv sync --frozen
    bash collect.sh          # git pull, analysis.py, proof_matrix.py, figures/figure_data.py

- `web/out/<tag>/raw/<cell>/<NNN>.json`: one run (system prompt, every reply with its reasoning, every action,
  `stopped_at_step`); `<NNN>.crash<k>.json` = a harness crash that was rerun with the same index and seed.
- `web/out/<tag>/access.log`: the server's own log (ground truth for follows); `job.log`, `manifest.jsonl`.
- `analysis.py` -> `results/analysis.json` (rates, bootstrap and seed-level intervals, status, prediction score) and
  `results/analysis_runs.csv` (one row per run with every event).
- `analysis.py --calib <parent web/out/gpt-oss-20b-d4>` -> `results/calib_parent.json`: the frozen rule on the parent
  run's logs (N 58/60, old warning 59/60, NV 0/60, CV 2/60 announcing followers).
- `CANARY_PORT=8821 uv run python web/test_harness.py`: no-model checks of the site, run order, prompts, stop rule.
