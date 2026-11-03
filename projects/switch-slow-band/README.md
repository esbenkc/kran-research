# switch-slow-band

Does the learned monitor switch in Qwen3-0.6B write into the attention sink's three slowest RoPE pairs (61-63)?
Room `theory-1103-switch-slow-band`. Frozen rules: `PREREG.md` (v1 + v1.1 s9); changes: `DEVIATIONS.md`.
Answer: **No** (REJECTED, both monitor copies; prereg outcome failed). No training, no model-API calls ($0).
PyTorch 2.14.1, transformers 5.20.2 (`env/` = rope-sink-band's pyproject + uv.lock), Mac mini M4.

## The copies and inputs (no new training)

Copied from sink-value (sha256 in DEVIATIONS D0): monitor s0, s1 (`numeric-chameleon-rl` r50 KL-1 policies at 8,192
updates; the STATUS copies) and grader s0, s1 (`stack-conditional`, reading rows only). All run the 2-block tail
(blocks 26-27 of Qwen3-0.6B, `src1025/tail.py`) on cached block-26 inputs: 200 test prompts x 34 tokens, cue-on
(recipient) and cue-off (donor). Tokens 0-2 reach the answer only through their keys and values in blocks 26-27.

RoPE pair j = head dims (j, j+64) (rotate_half), speed 1e6^(-j/64) rad per token; j = 63 is the slowest.
`pp.py` asserts the map on the model's own cos table.

## Files

| file | what |
|---|---|
| `patch.py` | sink-value's K/V patch engine + `kdims` (patch only some key dims), `alpha` (differentiable key path), mode `zero` (Phase 0: remove the q.k terms of a pair set) |
| `pp.py` | `Runner`: loads a copy, its inputs and donor keys; `px(spec)`, `share`, `grad_alpha`; `pair_arm(pairs)` |
| `gate.py` | engine gates: all-64-pair mask == whole-key patch bit for bit and == sink-value; autograd vs finite difference |
| `phase0.py` | PREREG s2: sink cut with pairs 61-63 zeroed (base tail, chat prompts), null cuts, per-head test of head 27.8, G = exact per-pair token-0 lead split; also writes the frozen null sets `data/sets.json` |
| `first_order_pairs.py` | (Iris) the frozen prediction `data/prediction.json` (PREREG s4, v1.1 s9) |
| `pairpatch.py` | PREREG s3 rows: G, BAND, COMP, S00-S63, U50, N200, E20, TG84, TOP3, HB, LOW8, SLOW36, W00-W61 |
| `analyze.py` | status (s5), CIs, nulls, law rows, prereg outcome -> `data/summary.json` |
| `proof_matrix.py` | `proof-matrix.md` from `data/summary.json` (keeps skeptic rows) |

## Data

- `data/patch/<copy>_<prec>_<dev>.json`: `P_on`, `P_off`, `rows {name: {pairs, P, share}}`, `per_prompt {name: 200
  P(X)}`, `top1 {name: 200 first-token ids}`. Status cells = `monitor_s{0,1}_fp32_cpu.json` (487 rows each).
- `data/phase0.json`, `data/sets.json`, `data/prediction.json`, `data/summary.json`.
- `ref/`: sink-value's frozen.json, summary.json, patch files (row-G targets); rope-sink-band's 0.6B `band.json`.
- `logs/`: every run with its `/usr/bin/time -l` peak.

## Rebuild (from the work dir; one job at a time, peak <= 2.5 GB)

```sh
PY=env/.venv/bin/python; cd kran-research/projects/switch-slow-band
$PY gate.py --copy monitor_s0                                                          # 9 s
$PY phase0.py                                                                          # 140 s
$PY first_order_pairs.py                                                               # 15 s
$PY pairpatch.py --copies monitor_s0,monitor_s1 --prec fp32 --dev cpu --rows status    # 440 s
$PY pairpatch.py --copies grader_s0,grader_s1 --prec fp32 --dev cpu --rows reading     # 125 s
$PY pairpatch.py --copies monitor_s0,monitor_s1 --prec bf16 --dev mps --rows reading   # 53 s
python3 analyze.py && python3 proof_matrix.py
```

Inputs not in git (`raw/`, `src1025/raw/cache/`) are copies of the sink-value inputs (sha256 in DEVIATIONS D0).
