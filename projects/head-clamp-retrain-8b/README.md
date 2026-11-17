# head-clamp-retrain-8b

At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?
Scale replication of `projects/head-clamp-retrain` (Qwen3-1.7B, branch swarm/theory-1107-near-tie-clamp) on
Qwen3-8B. Frozen plan: `PREREG.md`. Method credit: the probe build, the cached-tail training and the head unit follow
Sauers (github.com/SauersML/watching); the switch recipe and prompts are those of `projects/probe-switch-heads`.
PyTorch 2.14.1 (MPS), transformers 5.20.2, peft 0.21.2 (`pyproject.toml`, `uv.lock`). All local, no model-API calls.

## What runs

- `cfgm.py`: model constants from the HF config of `HC_MODEL` (default Qwen/Qwen3-8B: 36 blocks, 32 query heads,
  tail = blocks 34-35 = 64 heads, probe hook at block 18 = n_layers // 2, as 14 of 28 at 1.7B). `HC_OUT` = output dir.
- `step1_torch.py` (the parent's): cache of the inputs to the first tail block, probe off / on (stock
  `torch.compile`, read-only forward hook on the probe block's `q_norm`), 704 prompts of `inputs/prompts.jsonl`.
- `g0.py`: G0 = share of test prompts whose cached block-34 input at the answer position differs probe on/off;
  writes `data/g0.json` and `data/cache34_pos.npz` (answer position and position 71, fp32) for the readout forecast.
- `train.py --mode switch`: LoRA r16 on q,k,v,o,gate,up,down of the two tail blocks from scratch; T_f rule, 1.5 T_f
  save, cap 4,096. `train.py --mode clamp`: retraining from a switch adapter with a SWAP clamp, budget 2 T_f, stop at
  RETURNED. Checkpoint every 256 updates and at the box deadline; a restart resumes bit for bit.
- `audit_pick.py`: single-head SWAP audit (64 dev prompts), the frozen sets (SET, TOP-1, LOO, ALL-64, RANDOM 1-2),
  t=0 of every arm.
- `run.sh`: one laptop-queue box; skips finished cells, resumes cut runs, stops at the box deadline.
- `band_17b.sh` / `band_17b.py`: the 1.7B band, the parent's SET (and TOP-1) arms rerun on its adapters at the
  64-update dev grid (mini) -> `data/band_17b.json`.

## Engine gate (1.7B, mini)

`HC_MODEL=Qwen/Qwen3-1.7B HC_OUT=data/engine17b train.py --mode clamp --seed 0 --arm set --heads
27.7,26.5,26.0,26.14,27.12,27.3 --adapter inputs/parent17/lora_none_s0_2048_adapter.pt --steps 4096 --eval-every 128
--name set_s0_g128` on the parent's cache: RETURNED at 512 with all 512 per-update losses, grad norms and final
per-prompt rows equal to the parent's `set_swap_s0.json` (`logs/mini/engine_compare.txt`).

## Files

- `inputs/`: copies of the parent's prompts, its three 1.7B switch adapters and run files, its status.json.
- `data/runs/<name>.json`: args, t=0 scores, loss and grad norm per update, dev curve, final scores, per-prompt rows.
- `adapters/`: LoRA adapters of the 8B switch models and retrains (+ SHA256SUMS).
- `data/audit/s<seed>.json`: the first audit, the picked sets and the t=0 rows. `data/laptop/<job id>/`: box logs.
- `raw/` (not in git): the caches; SHA256 in `data/cache_sha.txt`.
