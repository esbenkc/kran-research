# nofollow-canary

Does the rel="nofollow" attribute on a stylesheet-hidden canary link cause Qwen3-8B's lower follow rate on raw HTML?
Room `defense-1124-length-or-markup`; follow-up to `anchor-text-law`. Frozen design and status rule: `PREREG.md`
(pushed c6173ca68, 2026-11-24T07:09:25Z); later facts: `DEVIATIONS.md`.

## Laptop-queue job (Esben's laptop, MLX)

| job | id | agent | runs | output | command |
|---|---|---|---|---|---|
| J1 | `tswdjfbrh2` | Qwen3-8B (`Qwen/Qwen3-8B` -> MLX 4-bit), thinking on | 15 gate + 5 x 90 + mdpad8 60 | `web/out/qwen3-8b/` | `bash labq_job.sh 2026-11-24T12:00:00-08:00 90` |

Harness code `922a47530`; `labq_job.sh` moves to the branch head and refuses to run if `web/`, `padding.txt`,
`pyproject.toml` or `uv.lock` differ from it. --mem-gb 12, --est-hours 4.5.

## Views (what the agent's fetch tool returns; `web/agent.py`)

raw (HTML bytes; the two hidden footer links carry `rel="nofollow"`), rawnf (raw minus that attribute), md
(markdownify 1.2.3), mdhtml (md with the two footer-slot links as inline `<a href>`), mdnf (mdhtml plus
`rel="nofollow"`), mdpad8 (md plus `padding.txt` before the footer, to 8x the raw page's Qwen3 tokens).

## Rebuild every number

    bash collect.sh        # git pull, test_analysis, analysis, proof_matrix, figure data

- `web/out/qwen3-8b/raw/<cell>/<NNN>.json`: one run (transcript, replies with reasoning, actions with the canary's
  character offset, answer, site ids); `access.log` (tab separated: time, ip, method, path, status, user agent, run
  label) is the only follow input; `job.log` (per run: follow, steps, seconds, peak GB); `manifest.jsonl`.
- `analysis.py` -> `results/analysis.json` (status, gates, legs, Newcombe and seed-level intervals, per-arm counts,
  secondary claims S1-S5) and `results/analysis_runs.csv` (one row per run).
- `CANARY_PORT=8824 uv run python web/test_harness.py`: no-model checks of the six views (586 OK,
  `results/test_harness.txt`, token counts in `results/precheck_tokens.csv`); `uv run python test_analysis.py`:
  boundary checks of the frozen rule (`results/test_analysis.txt`).
