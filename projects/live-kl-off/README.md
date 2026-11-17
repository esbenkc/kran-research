# live-kl-off

Does the 10% planted word take off by update 1,200 on the live (resampled-thought) engine when the thought KL penalty
is set to 0 and everything else is as in /on-policy-planted?

Qwen3-0.6B, bf16, PyTorch on MPS, Mac mini M4 (16 GB). All local, no model-API calls in the experiments ($0). Frozen
plan: `PREREG.md` (commit f144daf65, pushed 2026-11-17T01:00:18Z, before any data). Later changes: `DEVIATIONS.md`.
Parent: room oracle-1113-on-policy-planted, project `on-policy-planted` (branch swarm/oracle-1113-on-policy-planted).

## Engine

`live.py` is the parent's live trainer with ONE change (PREREG 'Engine'): the thought-KL row draw `rng.choice` runs
every update and is used only if `--kl-prefix > 0`, so the prompt order of seed 0 matches the parent's live_k4 run.
Two flags were added for the frozen schedule, neither changes a training update: `--stop-onset 0.30 --stop-confirm 2`
(end the seed after a read when the last 2 reads both have s_P >= 0.30; writes `onset_stop.json`) and a
`clock_stop.json` marker when `--stop-at` ends a run. The status arm runs `--kl-prefix 0`; the decision KL (0.1),
lr 3e-4, lam 0.42, k = 4 staleness, the cue (0.10 on ' We', token 1205) and the reads are the parent's defaults.

| file | what it does |
|---|---|
| `chain.py` | the frozen run tree: ' Let' placebo twin pool + its pilot read, seed 0, branch in code (TAKE-OFF -> seeds 1, 2; FLAT/EDGE -> positive control at cue 0.86, then seeds 1, 2), a placebo read after every seed; one heavy job at a time |
| `run_seeds.py` | one seed in process segments that end after the reads at 300, 600, 900 (resume from adapter + AdamW state); a watchdog kill resumes from the last saved read and moves later updates to `steps_killed_*.jsonl`, `batches_killed/` |
| `live.py` | the trainer; raw outputs in `data/runs/kl0_s<S>/`: `reads/c_<step>.npy` (cheat probability per stored-pool row), `reads.jsonl`, `steps.jsonl` (per update), `batches/<off|on>_<step>.npz` (every thought written), `adapter_<step>.pt`, `opt_<step>.pt` |
| `ctrl_pool.py`, `readpool.py` | placebo twin of the stored test pool (' Let' 6771 at thought token 0) and its reads -> `data/reads/let/c_<tag>.npy` |
| `kl_summ.py` | every number from the raw files (numpy only): `data/summary.json`, `analysis.json`, `figures/data.json`, `proof-matrix.md` |
| `inputs/` | the parent's inputs, byte-identical (sha256 in DEVIATIONS D5): prompts.jsonl, tokens.json, probe_v14.pt, pilot/adapter.pt (moved out of data/ after the run, D7; the run logs and run.json files still name their old data/ paths) |
| `data/parent/` | the comparison lines, copied from on-policy-planted: live_k4 (thought KL on), cached_d010_s0 (fixed thoughts), cachedkl_s0 (fixed thoughts + thought KL, exploratory, cut at 876) |
| `rebuild_pool.py`, `data/pool_cols/test/` | the stored test pool's committed token columns (cols.npz, meta.json, sha256 of the mini's h files); `rebuild_pool.py` recomputes its hidden states on another machine |
| `laptop_job.sh`, `env/` | the laptop-queue job for unsettled seeds (PREREG tree step 5): rebuilds the pool, its placebo twin and reads, checks the rebuilt pool against the mini's read 0, then runs each seed in full; `env/` pins the torch env (uv) |
| `watchdog.py` | footprint cap per job (9.0 GB) |
| `mt.py`, `cached.py`, `pool.py`, ... | the parent's model, pool and read code, unchanged |

## Rebuild

```sh
cd kran-research/projects/live-kl-off
# env: the parent's uv env (pyproject.toml + uv.lock of on-policy-planted): uv sync --frozen, at ../../../env/.venv
# the stored test pool data/pool/test (~0.8 GB) is not committed: copy it from on-policy-planted, or regenerate with
#   pool.py --mode test --R 14 --B 64 --X 1205 --out data/pool/test   (policy = inputs/pilot/adapter.pt); or rebuild_pool.py from data/pool_cols/test
/usr/bin/python3 chain.py --stop-at 20:00     # the whole frozen tree (placebo pool, seeds, branch)
/usr/bin/python3 kl_summ.py                   # every number, from the committed per-row reads
```
