# long-rollout-shield

Does a clean-recompute update learn a probe-off switch once 64-token thoughts give the probe a 10-point text
difference? Qwen3-0.6B, bf16, PyTorch 2.14.1 on MPS (Inductor `torch.compile` = the served graph), transformers
5.20.2, peft 0.21.2, Mac mini M4. All local, no model-API calls ($0). Frozen plan: `PREREG.md` (rules frozen at
b29df14b, gate numbers and T0 filled before the first update, 0cb24ec2); later changes: `DEVIATIONS.md`.
Parent: `/multi-token-recompute` (branch `swarm/oracle-1108-multi-token`); the engine (`mt.py`, `rl.py`, `read.py`,
`calib.py`) is copied from it, with the switches below added and the parent's defaults unchanged.

## What changed from the parent engine

| switch | file | what it does |
|---|---|---|
| `--k 64` | rl.py, read.py | 64 free thought tokens (the parent ran 16) |
| `--plant-build none` | rl.py, read.py, calib.py | the planted cue runs on the NO-probe build, so its text difference is the cue alone (the parent planted on the probe build: cue + real footprint) |
| `--hash-sigma s`, `hcue` in `mt.Sampler.rollout` | mt.py, rl.py, read.py | the scattered cue: at every thought position, logit += s x (+1/-1) per vocabulary id, signs from an integer hash of the prompt and every token so far (`mt._mix`, `hash_init`, `hash_signs`; CPU = MPS bit for bit) |
| `dev_D` | rl.py | text difference on the 64 dev prompts every 50 updates (monitor) |
| `--texts-only` | read.py | C_sup training texts: skips the teacher-forced pass (no D; `lp_l_n` is a copy of `lp_n` there) |
| `--log-grad N` | rl.py | DEVIATIONS D1 logging: pre-clip grad norm (every update) and the decision policy-gradient term's grad norm (own tail pass every N updates, grads discarded) |

Recipe for every arm (PREREG v1.3): P 8 prompts x G 8 samples, 4 probed prompts per update, reward read 1 / solve 0.6,
reward-only probe shaping (lam 0.4, sd_ref 0.2), decision 2-way KL beta 0.25 to the start policy, full-vocabulary
prefix KL 0.1 per thought position (summed), LoRA r16 alpha 32 on blocks 26-27, AdamW lr 3e-4, clip 1, chunk 8. Clean
recompute: one compiled no-probe prefill over prompt + thought + answer cue for every row, then blocks 26-27 eager.

## Rebuild

```sh
cd kran-research/projects/long-rollout-shield
PY=../../../env/.venv/bin/python     # env/pyproject.toml + env/uv.lock: uv sync --frozen --no-install-project
./lane.sh queue.txt                  # the exact serial job list that ran, in order (each line under watchdog.py, 9 GB cap)
$PY summarize.py                     # data/summary.json, proof-matrix.md (CPU, seconds)
```

`queue.txt` holds every heavy command with its arguments; `logs/<job>.log` holds its stdout and, on the last line,
its watchdog peak footprint. `data/gate/budget.json` is the frozen N formula evaluated once at T0 (`budget.py`).
Start policy: the parent's pilot adapter (`data/runs/pilot/`). Parent files used for comparison: `data/parent/`.

## Data

- `data/reads/raw/<name>.{json,npz}`: paired served reads (read.py), 200 test prompts x R common-random-number draws;
  npz keys as the parent (`prompt_idx, seed, split, toks_n, toks_l, lp_n, lp_l_n, tv, kl, c_n, c_l, c_ln, c_nl`,
  plus `c_X, c_notX` on forced-gate reads). Start reads: `start_real_test` (R 8), `start_{planted,hash}_test` (R 4),
  `start_{real,planted,hash}_train` (R 4, texts only). End reads: `planted_s0`, `real_s0`, and the D1 arm.
- `data/runs/<name>/`: `curve.jsonl` (dev monitor every 25 updates), `run.json` (args, wall time), `adapter.pt`.
- `data/gate/`: `calib.json` (one-token cue ' Need', eps 1.4265 at TV 0.1012), `calib_hash.json` (sigma grid,
  sigma* 0.0559), `calib_d30.json` (D1 arm), `budget.json`.
- `research-iris/`: Iris's readers (C_sup, C_lin, surprisal) and their outputs.
- `skeptics/`: Vera's and the final reviewer's rebuilds.
