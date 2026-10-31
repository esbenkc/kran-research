# massive-dims

Do a few massive-activation dimensions at the chat-template prefix carry both the grader switch and the monitor
switch? Qwen3-0.6B, PyTorch 2.14.1 (CPU and MPS), transformers 5.20.2, peft 0.21.2, Mac mini M4. No training, no
model-API calls ($0). Frozen plan: `PREREG.md` (v1-v1.2, all before any patch); changes: `DEVIATIONS.md`.

## The copies (no new training)

- **grader switch** s0, s1 (s2 extra): `stack-conditional` policies (blocks 26-27 LoRA, merged bf16, 8,192 GRPO
  updates). They say the flattery opener X more on the serving build (torch.compile) than on the grading build (eager).
- **monitor switch** s0, s1: `numeric-chameleon-rl` r50 KL-1 policies at 8,192 updates. They say X less when a
  leaky probe hook is attached. Merged here by `merge_r50.py` (on MPS, like the original rl.py; see DEVIATIONS D0).
- Both read CACHED block-26 inputs of the frozen, untrained trunk, so the block-26 input is the patch site and the
  untrained model's own state.

## What runs

- `src1025/cache.py --builds none,ref --passes rec --save-tail` re-caches the monitor inputs (the Oct 25 caches were
  deleted); gate G1 = logit sha256 equal to the Oct 25 files on 646/646 prompts.
- `merge_r50.py` merges the monitor adapters; gate G2 = cached P(X) equals `patch_1025.json`.
- `freeze.py` -> `data/frozen.json` (committed before any patch): massive set {1, 13, 35} (> 100x the median |x|
  at any of tokens 0-2, from all 4 base caches), R1 / R1-in / R1-pre energy predictions, sink positions.
- `massive.py --dev D --prec P --rows status|all` -> `data/patch/<copy>_<prec>_<dev>_<rows>.json`: P(X) on, off, and
  for each patch; share moved = (P_patched - P_on) / (P_off - P_on). Status rows (a) massive, (b) non-massive,
  (c) 99 random 3-dim sets rng(0), (d) all dims, at tokens 0-2. Reading rows R2-R6 (PREREG s2).
  Per-prompt P(X) for on, off, (a), (b), (d) is in `per_prompt`.
- `base_gap.py` -> `data/base_gap.json`: the untrained tail on the same inputs (control: no switch before training).
- `analyze.py` -> `data/summary.json` (frozen status rule, prereg outcome, reading rules) and `proof-matrix.md`.
- `chain.sh`: the J3 order (fp32 CPU all rows, bf16 CPU status rows, bf16 MPS all rows), one job at a time.

## Rebuild

```sh
cd ~/work/oracle-1031-massive-dims; PY=env/.venv/bin/python; P=kran-research/projects/massive-dims
$PY $P/src1025/cache.py --builds none,ref --passes rec --save-tail --out $P/src1025/raw/cache/base   # 74 s, 2.6 GB
$PY $P/merge_r50.py                                                                                  # 30 s
$PY $P/freeze.py                                                                                     # 13 s
$P/chain.sh                                                                                          # ~45 min, 1.2 GB
$PY $P/base_gap.py
python3 $P/analyze.py
```

Inputs not in git (`raw/`, see DEVIATIONS D0 for sha256): grader caches `raw/stack/cache/{eager,compiled}_rec_tail.npy`
and merged tails `raw/stack/runs/`, copied from oracle-1030-stack-conditional; monitor adapters `raw/r50/runs/`
from theory-1025-numeric-chameleon.
