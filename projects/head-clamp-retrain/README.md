# head-clamp-retrain

Does clamping a head set fixed before retraining stop a learned monitor switch from coming back?
Qwen3-1.7B, bf16, PyTorch 2.14.1 (MPS), transformers 5.20.2, peft 0.21.2, Mac mini M4 (16 GB). All local, no
model-API calls. Frozen plan: `PREREG.md` (v1 scout text quoted, v2 frozen at 48f25298 before any training update);
deviations with commit times: `DEVIATIONS.md`. Method credit: the probe build, the cached-tail training and the head
unit follow Sauers (github.com/SauersML/watching); the switched models, prompts and first audit are those of
`projects/probe-switch-heads` (branch swarm/oracle-1024-probed-only-switch).

## What runs

- `cache.sh`: rebuilds the parent's cache of block-26 inputs (stock `torch.compile`, read-only probe hook on block 14
  `q_norm` off/on, 704 prompts) with the parent's `step1_torch.py`; SHA256 must equal the parent manifest
  (`data/cache_sha.txt`: equal). Then `lm_states.py`: block-26 inputs of the base model on 64 held-out pile-10k
  documents (LM-loss text, `data/lm_meta.json`).
- `d79.py`: share of test prompts whose block-26 input at the answer position differs in any bit (`data/d79.json`: 1.000).
- `t0_gate.py`: forward only, no update. Each switched parent with each frozen clamp (`arms.json`) x kind
  (swap, umean, pmean, bmean): dev/test switch, push removed, LM loss; the PREREG v2 rule picks the clamp kind
  (`data/gate_t0.json`: PRIMARY = swap).
- `train_clamp.py`: warm start from the parent's LoRA adapter (r16, q,k,v,o,gate,up,down of blocks 26-27), clamp on
  from update 0, the parent's recipe (8 prompt pairs per update, AdamW lr 1e-4, grad clip 1.0, full-vocabulary CE
  at the answer position), 4,096 updates (2x the parent's 2,048), dev check every 128, early stop when dev strict
  switch >= 0.75 at 2 checks in a row (RETURNED). Final scores on the merged bf16 model with the clamp on.
- `clamplib.py`: the clamp. A head = one query head's 128-column slice of the o_proj input. SWAP (primary): in the
  run with the probe, the clamped slices get the detached slices of the run without the probe, same prompt, all
  positions.
- `audit_after.py`: WHERE audit of a retrained model: with its clamp on, each other head of 26-27 in turn gets a
  swap cut (push removed on 64 dev / 128 test prompts); plus the update kept only in attention or only in the MLP.
- `replication.py`: our ALL-32 s0 run vs the parent head-free cell hf_s0 (`data/replication_hf_s0.json`).
- `status.py` -> `data/status.json`; `proof_matrix.py` -> `proof-matrix.md`.
- `lane.py` runs `queue.txt` one job at a time under `watchdog.py` (8 GB cap); `logs/lane.log` has every start/end.

## Rebuild

```sh
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2, pyarrow>/bin/python   # here: ../../../env/.venv/bin/python
./cache.sh                       # ~3.5 min, peak 5.5 GB
$PY d79.py
$PY t0_gate.py --seeds 0,1,2 --kinds swap,umean,pmean,bmean   # ~6 min, 3.8 GB
$PY lane.py                      # every line of queue.txt in order (train_clamp.py / audit_after.py), ~0.33 s per update
$PY replication.py && $PY status.py && $PY proof_matrix.py
```
The exact arguments of every job are the lines of `queue.txt`; their start and end times are in `logs/lane.log`.

## Files

- `data/parent/`: the parent's adapters, run files and first audits (copied from probe-switch-heads), and the
  parent head-free run hf_s0 (probe-switch-audit-penalty).
- `data/runs/<arm>_swap_s<seed>.json`: args, clamped heads, loss and grad norm at every update, dev curve, t=0 and
  final scores, per-prompt rows (`rows`: [prompt, probe, top-1 id, logp(A), logp(B)]) at t=0, at every dev check and
  at the end, LM losses. `data/adapters/`: the retrained LoRA adapters.
- `data/t0/s<seed>.json`: the t=0 gate with per-prompt rows. `data/audit/`: WHERE audits.
- `raw/` (local, not in git): the caches; SHA256 in `data/cache_sha.txt` and `data/lm_meta.json`.
