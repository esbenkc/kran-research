# lora-switch-audit

Does the top input-side singular direction of a reward-trained LoRA delta rank the true cue first among 1,000 random
cue directions, in at least 5 of 6 trained switch models, with no behaviour test?

Organism: cue-size-law (Qwen3-0.6B, blocks 26-27 trained with LoRA r16 by group-relative policy gradient on cached
block-26 inputs; a planted cue at rows 0-2 = `<|im_start|>user\n` switches off a flattery opener). Code for the
organism is copied unchanged from `kran-research` branch `swarm/theory-1110-cue-size-learning`,
`projects/cue-size-law/` (tail.py, stack.py, watchdog.py, lane.sh); `rl.py` adds adapter snapshots only.
All local, PyTorch 2.14.1 on MPS / CPU, no model-API calls ($0). Frozen rules: `PREREG.md` (v1-v1.3).

## What runs

| step | file | output |
|---|---|---|
| block-26 input cache (from the parent run) | `stack.py` (parent's) | `raw/cache/eager_rec_tail.npy`, `meta.json` (copied from the parent, gitignored) |
| RL runs with adapter snapshots | `rl.py` via `lane2.sh A queueA.txt` and `lane2.sh B queueB.txt` (one job per lane, `watchdog.py` 2.5 GB) | `data/runs/<run>.json` (args, cue size x, dev curve, T), `raw/runs/<run>/step<N>.pt`, `logs/<run>.log` |
| parent end-of-run adapters | copied from the parent's `raw/runs/*_adapter.pt` | `raw/parent_runs/`, `data/parent_runs/*.json` |
| weight audit (CPU) | `audit.py` | `data/audit/<run>.json` (every module set x statistic: true score, rank, z, null p99, the 7 planted keys), `data/audit/<run>_null.npz` (1,000 null scores, qkv26) |
| known-answer check of the scorer | `known_answer.py` | `data/known_answer.json` |
| frozen clauses C1-C3, size curve, gain law, proof matrix | `summarize.py` | `data/summary.json`, `data/figure_data.json`, `proof-matrix.md` |
| theory (Felix) | `theory/*.py` | `theory/*.json` |

## Statistic

For block 26's q, k and v LoRA modules, dW_m = (lora_alpha / r) B_m A_m = 2 B_m A_m. Stack them (all three read the
same post-norm input) and take the top input singular vector v1. A cue u (3 x 1024, one unit row per template token)
at size alpha becomes c_p = N26(h_p + alpha |h_p| u_p) - N26(h_p) at rows p = 0, 1, 2, cast to bf16 as rl.py feeds it.
e1(u) = sum_p (v1 . c_p)^2 / sum_p |c_p|^2. Rank = 1 + #{null u' : e1(u') >= e1(u)} over 1,000 Gaussian u'
(numpy seed 20261112, rows unit-norm) planted the same way. Secondary (frozen): S3 gain = sum_p c_p^T G c_p / sum_p |c_p|^2
with G = sum_m dW_m^T dW_m; S2 A-only = sum_p sum_m |A_m c_p|^2 / sum_p |c_p|^2. Top-k energy for k = 2, 4, 8, 16
is exploratory (chosen after seeing data). At update 0 B = 0, so every score is 0 and all ranks are 1,001 (ties).

## Rebuild

```sh
cd kran-research/projects/lora-switch-audit
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python   # here: ../../../env/.venv (symlink)
# cache: copy raw/cache/{eager_rec_tail.npy,meta.json,real_rows012_u16.npy} from the parent project or rebuild with stack.py
./lane2.sh A queueA.txt & ./lane2.sh B queueB.txt &      # every run, snapshots in raw/runs/<run>/
NC_DEV=cpu $PY known_answer.py                           # scorer check
NC_DEV=cpu $PY audit.py                                  # data/audit/*.json (incremental; --force recomputes)
/usr/bin/python3 summarize.py                            # data/summary.json, proof-matrix.md
```

Training is deterministic on this machine: the reruns reproduce the parent's dev curves bit for bit (Vera, room
#20601). Do not run jobs with a python path inside another room's `~/work/<room>/` folder: `oracle.py watch` kills
those processes every 15 minutes after that room closes.
