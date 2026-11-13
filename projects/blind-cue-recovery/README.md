# blind-cue-recovery

Question (PREREG.md v2): Does subtracting no-cue adapters trained from the released adapter's own input matrix make the
top input direction of a reward-trained LoRA delta name the planted cue before the switch shows in behaviour, with no
suspect list?

Organism: cue-size-law / lora-switch-audit (Qwen3-0.6B, blocks 26-27 trained with LoRA r16 alpha 32 by group-relative
policy gradient on cached block-26 inputs; a planted cue at rows 0-2 switches off an opener token). The six cue models
`dir_u1110..u1115` (seed 0..5) and their snapshots are copied unchanged from `lora-switch-audit` (branch
`swarm/theory-1112-weight-only`). `rl.py` is the parent's file plus two flags (`--init-from/--init-mode`, `--no-eval`),
`audit.py` is the parent's file unchanged (the cue map and the 1,000 null cues). All local, torch 2.14.1 MPS/CPU, $0 API.

## What runs

| step | file | output |
|---|---|---|
| no-cue seeds 0-9, 256 updates | `rl.py` via `lane2.sh A queue.txt`, `lane2.sh B queue.txt` (`make_queue.py` writes the queue) | `raw/runs/nc_s<k>/step<t>.pt` |
| twins: A := target's A at t (SM), rowspace of B_t A_t (SMv), target's A_0 (SM0); B = 0; no cue; data seed = target seed + 100..103; t updates | same lanes | `raw/runs/<sm|smv|sm0>_<m0..m5|h6..h9>_t<t>_d<seed>/step<t>.pt` |
| scorer | `blind.py` (`Scorer.score`, `slice_null`, E1 `score_proj`) | |
| every frozen read, status, proof matrix | `analyze.py` | `data/summary.json`, `data/blind/cells.json`, `data/blind/nulls.npz`, `data/blind/slice_null.npz`, `proof-matrix.md` |
| cue-free diagnostic (before any cue read) + rebuild check | `diag.py` | `data/diag.json` |
| behaviour with bootstrap CI at 0, t_h, T | `beh.py` | `data/behaviour_ci.json` |
| known-answer checks | `check_slice.py`; init-from rebuild (README below) | `data/known_answer_slice_null.json`, `data/known_answer_init_from.json` |
| git copy of the tensors the reads need | `pack.py` | `data/adapters_qkv26/<run>/step<t>.pt`, `data/snapshot_sha256.json` |

Targets: m0..m5 = `dir_u1110_a0.03125_s0` .. `dir_u1115_a0.03125_s5`; h6..h9 = held-out no-cue `nc_s6..nc_s9`.
T (behaviour switch) = 32, 32, 32, 32, 24, 32; t_h (largest snapshot <= T/2) = 16, 16, 16, 16, 8, 16.

## Rebuild

```sh
cd kran-research/projects/blind-cue-recovery
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python
# From git alone (block-26 q, k, v tensors of every cell the status, phase 0, XS, SS, E1 and the SM curve read):
RUNS_DIR=data/adapters_qkv26 NC_DEV=cpu $PY analyze.py     # -> data/rebuild_packed/summary.json (SMv, SM0 cells absent)
# Full rebuild (needs raw/cache from the parent project; training is deterministic, bit for bit):
python3 make_queue.py > queue.txt; ./lane2.sh A queue.txt & ./lane2.sh B queue.txt &
NC_DEV=cpu $PY diag.py; NC_DEV=cpu $PY analyze.py; $PY beh.py; $PY pack.py
```
Checks that passed: `--init-from <target step0> --seed <target seed>` rebuilds `dir_u1110` and `c2_a0_s1` at 16 and 32 updates
bit for bit; `--no-eval` runs rebuild the parent's `c2_a0_s0/s1` at all 16 shared snapshots bit for bit (`data/diag.json`);
every raw snapshot's sha256 is in `data/snapshot_sha256.json`.
