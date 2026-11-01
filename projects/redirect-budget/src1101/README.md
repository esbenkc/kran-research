# sink-value

Do sink-heavy attention heads carry the grader and monitor switches in the values they copy? Qwen3-0.6B, PyTorch
2.14.1 (CPU and MPS), transformers 5.20.2, Mac mini M4. No training, no model-API calls ($0). Frozen plan:
`PREREG.md` (v1-v1.2, all before any patch); changes: `DEVIATIONS.md`. Builds on `massive-dims` (same copies, inputs
and read-out; its row (d), the full residual patch at tokens 0-2, is this project's gate G0).

## The copies (no new training)

- **grader switch** s0, s1: `stack-conditional` policies (LoRA on blocks 26-27, merged bf16, 8,192 GRPO updates).
- **monitor switch** s0, s1: `numeric-chameleon-rl` r50 KL-1 policies at 8,192 updates (MPS merge).
- All four read CACHED block-26 inputs of the frozen, untrained trunk. Tokens 0-2 of the 34-token chat prompt reach
  the answer token only through their keys (K) and values (V) in the attention of blocks 26 and 27.

## What runs

- `freeze.py` -> `data/frozen.json` (before any patch): sink-heavy query heads (mean attention on tokens 0-2 over
  query positions 3-33 > 0.5; base tail; condition-on input; CPU fp32) = 29 of 32; the 99 status random sets
  (rng(0)); all 1,920 size-matched sets (R4); the non-sink heads {26.9, 27.7, 27.13}; per-head masses.
- `patch.py`: the engine. A registered attention function that, for chosen QUERY heads, swaps in the
  condition-off run's K (after k_norm and RoPE) and/or V at chosen key positions, at query positions >= 3 (R7: the
  answer position only). Donor K/V come from a capture pass of the same copy on the condition-off input. Unpatched
  heads keep the stock sdpa output bit for bit.
- `run_patch.py --dev D --prec P --rows status|reading|r1|r4` -> `data/patch/<copy>_<prec>_<dev>_<rows>.json`:
  P_on, P_off, per row P and share, and per-prompt P(X) (200 values) for every row.
  share moved = (P_patched - P_on) / (P_off - P_on), the whole switch.
- `r6_split.py` -> `data/r6_split.json`: exact per-head split of the answer-token head output change into a V-term,
  a K-term on token 0 (the sink's value, re-weighted), token 1, token 2 and tokens 3-33, weighted by dP(X)/d(head
  output at the answer position).
- `attn_onoff.py` -> `data/attn_onoff.json`: per head, answer-token attention on tokens 0, 1, 2, cue on vs off.
- `r9_diag.py` -> `data/r9_diag.json`: per head, token-0 value norm / mean value norm of tokens 3-33, the same
  through the output projection, and the effective rank of head outputs; base and trained copies.
- `analyze.py` -> `data/summary.json` (gates, bootstrap intervals, frozen status rule, prereg outcome) and
  `proof-matrix.md`.
- `chain.sh` (ran step 1 only; its bash died), `chain2.py`, `chain3.py`: the job order, one job at a time (D3).

## Rebuild

```sh
cd ~/work/oracle-1101-sink-value; PY=env/.venv/bin/python; P=kran-research/projects/sink-value
$PY $P/freeze.py                       # 14 s, 1.9 GB
for r in "cpu bf16 status" "cpu fp32 status" "mps bf16 reading" "mps bf16 status" "cpu fp32 reading" "cpu bf16 r1"; do
  set -- $r; $PY $P/run_patch.py --dev $1 --prec $2 --rows $3 --threads 6   # CPU bf16 status ~25 min; others 2-10 min
done
$PY $P/run_patch.py --dev mps --prec bf16 --rows r4 --copies grader_s0        # ~20 min
$PY $P/r6_split.py; $PY $P/attn_onoff.py; $PY $P/r9_diag.py
python3 $P/analyze.py
```

Inputs not in git (`raw/`, `src1025/raw/cache/`; sha256 in DEVIATIONS D0) are copies of the massive-dims inputs.
