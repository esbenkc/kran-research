# Vera (skeptic) sign-off — theory-1031-refusal-neuron / refusal-neuron-massive

Fresh context; I did not read Bruno's code (run.py, arms.py, analyze.py). Every number below comes from my own scripts in
this folder, run on the raw per-prompt files in `data/runs/` and on the model weights. Definitions: PREREG v1 @1ac9925,
DEVIATIONS D1-D11.

## 1. Rebuilds from raw data (status cell + replication cell)
Script `vera_check.py` (labels re-derived from reply text with `vera_labels.py`; distinct ratio from token ids;
D5 garbage threshold = p95 of clean-harmless neutral-turn NLL = 3.0966; paired bootstrap 10,000, seed 0, percentile).

| cell | label | EFFECT A | EFFECT B | EFFECT C | r_keep [95% CI] | r_rem [95% CI] | rule |
|---|---|---|---|---|---|---|---|
| TEST 200 | string (status) | +0.315 [0.235,0.400] | +0.320 | +0.015 [-0.010,0.040] | 0.048 [-0.033,0.136] | 1.016 [0.946,1.096] | REJECTED |
| TEST 200 | judge (D3) | +0.435 [0.360,0.510] | +0.445 | +0.000 [-0.015,0.015] | 0.000 [-0.034,0.034] | 1.023 [0.935,1.122] | REJECTED |
| DEV 100 | string | +0.540 [0.44,0.64] | +0.550 | +0.020 [0.00,0.05] | 0.037 [0.000,0.096] | 1.019 [1.000,1.064] | REJECTED |
| DEV 100 | judge | +0.460 [0.36,0.56] | +0.450 | +0.000 | 0.000 [0,0] | 0.978 [0.854,1.111] | REJECTED |

Phase 0: TEST string refusal base 0.460 -> A 0.025 (-43.5 pts >= 30). Harmless 200: garbage A/B/C 6.5/5.5/5.0% vs base
5.0%, harmless refusal 0% in every arm -> coherence passes. Gate vs random neurons (D6, first-token): see section 3.

## 2. Independent forward pass (own code, no HF modeling)
`vera_fwd.py`: Qwen3 decoder written from the config (RMSNorm, q/k-norm, RoPE theta 1e6, GQA 16/8, SiLU MLP), CPU fp32,
weights streamed from safetensors by seek+read; the pin as out_mlp += (m - h_3270(t)) * v.
- massive dims (CHAN 32, L13 output, token 0): 1793/1999/1401 = 14614/9028/908; max |value| at any other position 126.
- first-token refusal log-odds ({I, Sorry}) on TEST 0-49 vs Bruno's saved scores: per-prompt corr 0.9991-0.9997 for
  base/A/B/C/A_tok0/N5441pos; drop vs base mine/his: A 9.588/9.640, B 9.628/9.661, C -0.048/-0.048,
  A_tok0 -0.024/+0.020, N5441pos 1.346/1.393 (`vera_cmp_ft.py`).
- greedy text, base, TEST 0-49: first token same 49/50, first 8 tokens 46/50, refusal 0.42 vs 0.44 (`vera_cmp_own.py`).
- D5 garbage NLL rescored with my own teacher-forced pass (`vera_nll.py`), 40 harmless-base + 40 A replies:
  corr 0.9999 / 1.0000, mean |diff| 0.011 / 0.010 nats.
- weight facts (`vera_static.py`): ||w_3270|| = 1.1085 (layer median 1.465); energy share in M 0.002703 (rank 3126/6144,
  layer median 0.00280, chance 3/2048 = 0.00146); best 3 coords of ANY kind hold 1.68% of energy (13% of length), so
  prereg clause 1 (>= 50%) fails by construction.

## 3. Null tests
(filled below)

## 4. Definition-stretch checks
- String refusal list vs LLM judge (D3; `vera_judge.py`, claude-haiku-4-5, cached): agreement only 56-75% per arm; the
  list misses lecture replies ("X is a serious crime ... not only illegal"). Status agrees under both labels.
- N5441 row: string EFFECT +12.5 pts is a regex artifact; judge EFFECT -1.0 [-4.5,+2.5]; 37/40 string flips are reworded
  refusals.
- Both labels see only the first 64 tokens; 2605.08513 judged 512 tokens, so absolute compliance rates are not
  comparable to the paper's ASR.

## 5. Novelty
See novelty.md.

## Verdict
(filled below)
