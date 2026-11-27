# reach-law-gpt-oss

Does one reach rule predict decoy-key use for gpt-oss-20b? Pre-registration: `PREREG.md` (stamped 07:08:08Z);
changes: `DEVIATIONS.md`.

## Layout

- `shell/env.py`: the sandbox (route-first-law shell env; arms none / runbook / linked byte-equal to the parent, plus
  prompt). `WorkspaceEnv.reach()` logs reach and key use per run.
- `shell/agent.py`: agent loop (parent prompt, step cap 20); `shell/mlxcall.py`: local MLX backend (anchor-text-law).
- `run.py`: the resumable job (gate, main, prompt, filler); `labq_job.sh`: the laptop-queue command.
- `test_arms.py`: harness checks, no model (parent byte check, reach and key-use edges).
- `analysis.py`: frozen rule (PREREG s3), secondary analyses (s4), prediction scoring (s5), from raw files only; it
  re-derives reach and key use with its own code and exits 2 on any mismatch with `env.reach()`.
- `proof_matrix.py`: `proof-matrix.md` + one JSON per cell in `results/cells/`.
- `smoke/qwen3-1.7b/`: 4 non-counted plumbing runs (env seed 800).

## Continuation (after laptop job ka12ju3i4s lands)

```
git pull --rebase --autostash                # the labq worker commits data/gpt-oss-20b/
uv sync --frozen
uv run python test_arms.py                   # ALL PASS
uv run python analysis.py                    # results/analysis.{json,txt}, analysis_runs.csv; exit 0
uv run python proof_matrix.py                # proof-matrix.md, results/cells/
uv run python figures/build_main.py          # main figure data (Selma)
```

Check first in `data/gpt-oss-20b/job.log`: the `start` line (`effective=` stop time), parse counts per run (`parse=`),
`caps=` (reasoning cap hits), `peakGB`; and `data/gpt-oss-20b-test_arms.txt` (harness tests on the laptop clone; the
parent byte check is skipped there when the parent branch is absent). Then the skeptic runs
`skeptics/skeptic/vera_check.py` and `vera_nulls.py` on the same raw files and compares with `results/analysis.json`.
