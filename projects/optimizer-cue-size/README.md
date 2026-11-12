# optimizer-cue-size

Does plain SGD learn a cue-conditional switch with a cue-size exponent at least 0.6 steeper than AdamW, while
sign-SGD matches AdamW? Room theory-1112-optimizer-sets. Qwen3-0.6B, PyTorch 2.14.1 on MPS (transformers 5.20.2,
peft 0.21.2), Mac mini M4. All local, no model-API calls ($0).
Frozen plan: `PREREG.md` (v1 84ee6d172, v1.1 8a7288ea2). Every change after the freeze: `DEVIATIONS.md` (D0-D6).
Organism: the parent's (cue-size-law, origin/swarm/theory-1110-cue-size-learning) first-token GRPO bandit on cached
block-26 inputs with a planted cue at rows 0-2, unchanged except the optimizer and the flags below.

## What runs

| step | file | output |
|---|---|---|
| block-26 input cache (copied from the parent, regenerate with the parent's `stack.py`) | - | `raw/cache/eager_rec_tail.npy`, `raw/cache/meta.json` (not committed, 45 MB) |
| fp32 cue size x per alpha (step 0 only) | `rl.py --step0-only` | `data/step0/x_a*.json`, `data/x_fp32.json` |
| every RL run | `rl.py` via `lane.sh A queue.txt` (one lane, each job under `watchdog.py --cap-gb 2.5`, peak 1.76 GB) | `data/runs/<run>.json` (args, cue size, all evals with per-prompt P(X), text health, step-1 norms, per-window grad norm, clip fraction, ||A||, ||B||, gradient along the cue-read direction), `logs/<run>.{log,cmd}`, `raw/runs/<run>_adapter.pt` (not committed) |
| learning-rate picks (PREREG s8.1 + D1, D2 v2, D4) | `pick_lr.py --arm sgd|adamwtm|sgdfa` | `data/{sgd,adamwtm,sgdfa}_pick.json` |
| frozen analysis (PREREG s6) + secondaries | `analyze.py` (numpy + scipy) | `data/summary.json`, `data/figure_data.json`, `proof-matrix.md` |
| mechanism read (gradient SNR along the cue-read direction per quarter of T) | `mech.py` | `data/mech.json` |

`run.py` runs the last three steps in order (about 2 s).

## rl.py flags added in this room

- `--opt adamw|sgd|sign`: AdamW (betas 0.9/`--beta2`, wd 0, grad-norm clip 1.0), plain SGD (no momentum, no clip),
  sign-SGD (`p -= lr * sign(g)`, no momentum, no clip). `--clip` overrides; the clip-active fraction is logged.
- `--freeze-a`: only LoRA B trains (the A-frozen arm).
- `--dense 4:256,16:2048,64`: eval schedule (PREREG s3). T = first eval with D >= 0.10 at two evals in a row, read
  on the union of `dev_curve` and `dense_curve` (`T_dense` in the JSON; the `T` field is the parent's coarser grid
  and is used only for Phase 0).
- Text health every 25 updates; a run stops if cue-off entropy < 0.5x step 0 or top-1 mass > 0.9 at 2 checks in a
  row (PREREG s8.4 with D1). Early stop after a switch at max(1.5 T, T + 256).
- Cue-read direction d = unit gradient of the dev gap over the trained LoRA weights, recomputed at every parent-grid
  eval; per update the raw training gradient is projected on d (`proj_mean`, `proj_sd` per window).

## Rebuild

```sh
cd kran-research/projects/optimizer-cue-size
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python
# cache: copy raw/cache/{eager_rec_tail.npy,meta.json} and raw/prompts.jsonl from the parent, or rebuild with its stack.py
./lane.sh A queue.txt          # every run in queue order (~2 h on an M4); queue lines are `<name>|<rl.py args>`
/usr/bin/python3 run.py        # picks, frozen fit, matrix, figure data, mechanism read
```

Runs are deterministic on the same machine: the Phase 0 rerun of the parent's AdamW x 33.3 seed 0 reproduced the
parent's dev gap at every eval to 5 decimals (Vera #20314).
