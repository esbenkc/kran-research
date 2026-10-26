# probe-switch-audit-penalty

Does a probe-triggered switch keep firing when training also hides it from a head audit?
Qwen3-1.7B, bf16, PyTorch 2.14.1 (MPS, `torch.compile` / Inductor), Mac mini M4 (16 GB). All local, no model-API calls.
Frozen plan: `PREREG.md`; changes with commit times: `DEVIATIONS.md`. Follows `projects/probe-switch-heads` (same
probe, same prompts, same cache, its three trained switches are the warm starts). Method credit: the probe build,
the cached-tail training and the head unit follow Sauers (github.com/SauersML/watching).

## What runs

- Cache (`chain_j1.sh` -> `step1_torch.py`): block-26 inputs with and without the leaky q_norm probe (block 14),
  704 prompts x 80 tokens, stock compile. Rebuilt here; sha256 equals the probe-switch-heads cache
  (none 58557064..., probe 7aae94d7...; that project's data/manifest_raw.json).
- Warm starts (`data/warm/s{0,1,2}_adapter.pt`, copies of probe-switch-heads `data/L/runs/lora_none_s0_2048|s1|s2_adapter.pt`;
  sha256 56fbff28..., b05450a4..., e50e9ef3...).
- Training (`train_pen.py`, penalty code in `pen.py`): LoRA r16 on blocks 26-27, lr 1e-4 AdamW with a 32-update
  linear warmup (DEVIATIONS D4), 8 pairs per update in micro-batches of 4, CE on A (probed) / B (unprobed) +
  lambda * ln2 * P / P0. P = top-3 single-head blind removal + all-32 blind removal (KL nats, PREREG). Block-27
  heads exact (answer position only), block-26 heads 4 random per micro-batch. `--headfree` = arm A0 (all 32 head
  slices swapped to the unprobed run in every training pass, CE only). `--steps 0` = merge the warm start only.
  Dev monitor every 64 updates (fast path in pen.py): switch, push, per-head removal, all-32 removal, switch with
  all 32 heads swapped (D2 cell). CE vs penalty gradient norms at updates 1-20 and every 64.
- Whole-graph eval (`eval_whole.py`, unchanged from probe-switch-heads): merged tail in the full model, one compiled
  graph per run and condition, no recorder, 128 test prompts, probe on/off.
- Audit (`audit.py`, Iris): exact hook patching on the cached tail; blind greedy top-3 on dev scored on test (title
  cell), informed, cut-moves-output, R(k), all-32, reverse swap, nulls, identity swap, MLP path patches, clean KL.
- `chain.sh`/`chain2.sh`: one job at a time from `queue.txt` under `watchdog.py`; `slot.sh` gives a teammate an
  exclusive slot between jobs. `summary.py`: data/summary.json + proof-matrix.md.

## Rebuild

```sh
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python   # here: ../../../env/.venv/bin/python
./chain_j1.sh                                                   # cache, ~3 min, 5.4 GB
$PY train_pen.py --name hf_s0 --seed 0 --warm data/warm/s0_adapter.pt --headfree --warmup 32 --steps 512
$PY train_pen.py --name lam100_s0 --seed 0 --warm data/warm/s0_adapter.pt --lam 100 --warmup 32 --steps 512
$PY train_pen.py --name warm_s0 --seed 0 --warm data/warm/s0_adapter.pt --steps 0
$PY eval_whole.py --runs hf_s0,warm_s0,lam100_s0 --emulate 0
$PY audit.py --run lam100_s0 --ref raw/step2/Qwen3-1.7B/runs/warm_s0_merged.pt
$PY summary.py
```
The exact order and arguments of every job are in `logs/chain_copy.txt` (START lines; copy of the live logs/chain.log).

## Runs (all 512 updates from the warm start of the same seed unless said)

| name | what | seed |
|---|---|---|
| warm_s0, warm_s1, warm_s2 | the probe-switch-heads adapters, merged only (`--steps 0`) | 0, 1, 2 |
| hf_s0, hf_s0_2048 | A0 head-free by construction; `_2048` continues hf_s0 to 2048 updates (fresh Adam + warmup) | 0 |
| lam100_s0, lam100_s1 | weight 100 (the goal's 100x); lam100_s2 started and was stopped after 25 s (Felix #7278), not counted | 0, 1 |
| lam10_s0, lam10_s1 | weight 10 | 0, 1 |
| lam1_s0 | weight 1 (n=1, D8) | 0 |
| lam0_s0 | weight 0: the control (same updates, no penalty) | 0 |
| X_s0 | weight 100 with all 32 heads exact in the penalty every update (`--k26 16`, DEVIATIONS D7) | 0 |

Continuation and X commands, as run (from logs/chain_copy.txt):
```sh
$PY train_pen.py --name hf_s0_2048 --seed 0 --warm raw/step2/Qwen3-1.7B/runs/hf_s0_adapter.pt --headfree --warmup 32 --start-step 512 --steps 1536
$PY train_pen.py --name X_s0 --seed 0 --warm data/warm/s0_adapter.pt --lam 100 --k26 16 --warmup 32 --steps 512
```
Incidents (also in logs/chain_copy.txt): 18:27 PDT a mis-armed slot.sh started a second chain for ~20 s (two warm
merges overlapped; no scored number depends on it); two skeptic/research jobs overran their slots next to a training
run (room #7313, #7315). None of these change a stored number: every job is its own process with fixed seeds.

## Files

- `raw/` (local, not in git): the cache, merged tails, adapters, optimizer states, per-run JSONs.
- `data/` (git): `runs/<name>.json` (args, P0, dev curve with the monitor, grad log, cached test rows),
  `whole/<name>.json` (per-prompt whole-graph rows), `audit/<name>.json` (audit.py), `summary.json`, `warm/`.
- `logs/`: every job's log with its watchdog peak footprint; `smoke_lam100.log` = the uncounted 20-update smoke
  without warmup (DEVIATIONS D4).
