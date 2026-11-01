# probe-switch-frozen-heads

Does a probe-triggered switch need a trainable attention head?
Qwen3-1.7B, bf16, PyTorch 2.14.1 (MPS, `torch.compile` / Inductor), transformers 5.20.2, Mac mini M4 (16 GB). All
local, no model-API calls. Frozen plan: `PREREG.md` (v1-v1.4, commits 0491fcb..104b1f5, all before any MLP run);
changes with commit times: `DEVIATIONS.md`. Same probe, prompts and splits as `projects/probe-switch-heads` and
`projects/probe-switch-audit-penalty` (branches swarm/oracle-1024-probed-only-switch, swarm/oracle-1026-audit-penalty).
Method credit: the probe build, the cached-tail training and the head unit follow Sauers (github.com/SauersML/watching).

## What runs

- Probe: forward hook on block 14 `self_attn.q_norm` that keeps a reference to the output (read-only). With stock
  `torch.compile` this hook changes the compiled graph, so later activations differ in their last bits (the leak).
- `cache.py --pass A`: whole compiled graph, one recorder (pre-hook) at the block-20 input, all 704 prompts, probe
  off/on -> `raw/cache/{none,probe}_A_b20.npy` (uint16 bf16 bits [704, 80, 2048]) + answer logits. The 8-block arms
  train on exactly these inputs. `--pass B`: recorders at every block input / mid-block 14..27 + final norm,
  positions 71 and 79 only (a different compiled graph: the recorders move the bits; PREREG v1.3).
- `research_checks/map_states.py` (Iris): the base eager tail (blocks 20-27) on the pass-A inputs -> states for the
  readability map and `raw/cache/{none,probe}_eager_mid27_79.npy` (block-27 post-attention residual at the answer
  token; the last-MLP arm trains on it). `research_checks/map_probe.py` -> `map.json`; route -> `route.json`.
- `train.py --arm all8|mlp8|mlp27 --seed S --steps N`: LoRA r16 alpha 32 (own 20-line LoRA module, same math as
  peft Linear with fp32 adapters), lr 1e-4 AdamW, 8 prompt pairs per update in micro-batches of 4, grad clip 1.0,
  full-vocabulary CE at the answer position (probed -> 'A', unprobed -> 'B'). all8 = q,k,v,o,gate,up,down of blocks
  20-27; mlp8 = gate,up,down of blocks 20-27 (every attention weight frozen); mlp27 = gate,up,down of block 27 on the
  cached block-27 post-attention answer vector. Logs loss, pre-clip grad norm and RMS grad per parameter at EVERY
  update; dev strict switch every 128; early stop at dev strict >= 0.95 twice. Merged into bf16 at the end and scored
  on test (cached tail). `--smoke` = speed smoke only (no test score).
- `eval_whole.py --runs base,<run>,...`: title score. Merged bf16 weights in the full model, one compiled graph per
  condition, no recorder, 128 test prompts, probe on/off, a second pass for determinism, SHA256 of the logits.
- `audit.py --run <8-block run>`: eager merged tail on cached inputs; head-clamp of all 128 heads of blocks 20-27,
  per-block clamps, blind greedy top-3 (dev pick, test score, R_blind3), 50-set random null; saves after-training
  states at positions 71/79 for `after_map.py` (CPU logistic AUROC per site, 200-flip null).
- `summary.py`: `data/summary.json` + `proof-matrix.md`.
- `chain.sh`: one job at a time from `queue.txt` under `watchdog.py` (phys_footprint cap 8.5 GB, bound to the child pid).

## Runs (from scratch; early stop at dev strict >= 0.95 twice; title score = whole compiled graph, 128 test prompts)

| name | arm | seed | updates | T75 | whole-graph strict |
|---|---|---|---|---|---|
| all8_none_s0 | all-modules, blocks 20-27 (control) | 0 | 896 | 768 | 1.000 |
| all8_none_s1 | all-modules, blocks 20-27 (control) | 1 | 896 | 768 | 1.000 |
| mlp8_none_s0 | MLP-only, blocks 20-27, heads frozen | 0 | 1152 | 1024 | 1.000 |
| mlp8_none_s1 | MLP-only, blocks 20-27, heads frozen | 1 | 1152 | 1024 | 1.000 |
| mlp8_none_s2 | MLP-only, blocks 20-27, heads frozen | 2 | see data/summary.json | | |
| mlp27_none_s0, _s1 | last-MLP (block 27 MLP only) | 0, 1 | 2048 | none | 0.000, 0.000 |
| mlp27_shuffled_s0 | last-MLP, shuffled labels (null) | 0 | 2048 | none | 0.000 |
| mlp27_none_s0_8k | last-MLP, 8192-update budget | 0 | see data/summary.json | | |

ALL-8 s2 did not start: its full budget could not end by the PREREG v1.2 clock cut (21:15 PDT), so the control is n = 2.
Audits (`data/audit/`): MLP-8 s0 and s1 (clamp, blind top-3, LoRA split, states), ALL-8 s0 (clamp + states only).

## Rebuild

```sh
PY=<venv with torch 2.14.1, transformers 5.20.2, scikit-learn 1.9.1>/bin/python   # here: ../../../env/.venv/bin/python
$PY cache.py --pass A && $PY cache.py --pass B                                      # ~3 + ~3 min, peak 5.1 GB
$PY research_checks/map_states.py states . raw/map && $PY research_checks/map_probe.py . raw/map research_checks/map.json
$PY train.py --arm all8 --seed 0 --steps 2048
$PY train.py --arm mlp8 --seed 0 --steps <N per PREREG>
$PY train.py --arm mlp27 --seed 0 --steps 2048
$PY eval_whole.py --runs base,all8_none_s0,mlp8_none_s0
$PY audit.py --run mlp8_none_s0 && $PY after_map.py --run mlp8_none_s0
$PY summary.py
```
The exact order and arguments of every job are in `logs/chain_copy.txt` (START lines).

## Files

- `raw/` (local, not in git): caches, states, adapters. SHA256 in `data/manifest_raw.json`.
- `data/` (git): `runs/<name>.json` (args, dev curve, grad_log of every update, cached test rows with top-1/top-2
  ids and logits, logit and logp of A and B), `whole/<name>.json` (per-prompt whole-graph rows), `audit/`, `after_map/`,
  `checks/`, `summary.json`, `prompts.jsonl`.
- `logs/`: every job's log with its watchdog peak footprint; `*.before1806` = cache attempt 1, which ran from a closed
  room's venv and was stopped by the oracle's leftover-process cron (rc 143, nothing used from it).
