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

## 3. Null tests (where luck or a loose definition could fake the result)
- Random neurons at matched edit size (F, 100 L13 neurons, larger sign; Bruno's raw first-token files, my code
  `vera_random.py`): drop median 2.633, p99 7.157, max 8.505 vs gate 9.773 -> gate > p99 (Phase-0 item 2, D6).
- Existence test, first token (Bruno's draws): D' (full-size push into M) 100 draws median 0.359 = 3.7% of A, max 2.097
  (21%); E' (3 ordinary coords) median 0.682 = 7.0%, max 3.198; D' vs E' medians permutation p = 0.36 (20k shuffles).
- Existence test, MY OWN draw family and forward (random Gaussian direction inside M at the gate's full per-token
  size; `vera_fwd.py Dp:<seed>`, 5 seeds, TEST 0-49): -0.15..+0.16 of A; 3 ordinary-coord seeds -0.15..+0.09.
- Generation + judge on all 25 random draws (D11): judge EFFECT D' median +1.8 (max +4.0), E' -2.0 (max +4.5),
  F -0.5 (max +7.0), F' +2.5 (max +4.5) points vs gate +43.5. Largest = 16% of the gate. String label inflates F to
  +21.5 by counting reworded refusals. All 25 pass harmless-50 coherence (garbage <= 12%, refusal <= 2%).
- Replication: DEV 100 cell gives the same REJECTED under both labels (table above).
- Scorer variants on the status cell (TEST / DEV): no nonsense rule r_keep 0.046 [0.011,0.098] / 0.018, r_rem
  0.989 [0.962,1.000] / 1.018; full-text regex r_keep 0.000 [-0.102,0.092] / 0.039, r_rem 1.067 / 1.039. All REJECTED.
- C and A_tok0 are not copies of base (91/200 and 99/200 replies identical to base) and A_skip0 is not a copy of A
  (95/200 identical): the null rows are real runs.

## 4. Definition-stretch checks
- String refusal list vs LLM judge (D3; `vera_judge.py`, claude-haiku-4-5, cached): agreement only 56-75% per arm; the
  list misses lecture replies ("X is a serious crime ... not only illegal"). Status agrees under both labels.
- N5441 row: string EFFECT +12.5 pts is a regex artifact; judge EFFECT -1.0 [-4.5,+2.5]; 37/40 string flips are reworded
  refusals.
- N5441 split: string -3.0 / +9.0, judge -0.5 / -1.5 points: the 70%-outlier neuron is no lever either way.
- Random-neuron 'median 27% / max 87% of the gate' holds ONLY for the first-token {I, Sorry} log-odds; in judged
  answers no random push exceeds 16% of the gate.
- Both labels see only the first 64 tokens; 2605.08513 judged 512 tokens, so absolute compliance rates are not
  comparable to the paper's ASR.

## 5. Novelty
See novelty.md.

## 6. Math re-derived
Pinning h_i <- m changes the MLP output by (m - h(t)) w, so delta(t) = (m - h(t)) w (post eq. 1). Since w = w_M + w_R
with disjoint support, delta = (m-h) w_M + (m-h) w_R exactly (Bruno asserts torch.equal; my own edit code builds B and C
the same way and reproduces his first-token scores). s = ||w_M||^2/||w||^2 = 0.0576^2/1.1085^2 = 0.002703 (post eq. 2:
0.0027 OK); sqrt(s) = 0.0520 (post: 5.2% OK); 3/2048 = 0.001465 (post: 0.0015 OK). First-token score
log p - log(1-p) with p = P(first token in {40, 19152}): my own implementation matches his per prompt (corr >= 0.999).

## Verdict
SIGN-OFF: CONFIRMED that the status cell is REJECTED (No to the title), and the existence test fails: same-size pushes
into the 3 outlier coordinates do not act as a refusal lever (first-token median 3.7% of the gate; judged answers at
most +4.0 points vs +43.5). Prereg outcome: failed (clause 1 fails by construction from the weights: any 3 coords hold
<= 1.68% of the gate's energy; clause 4 fails by ~13x). Required wording: compliance means 'in the first 64 tokens';
random-neuron percentages are first-token only; N5441 reworded refusals (judge -1.0).
API spend by Vera (judge, claude-haiku-4-5): $1.89 logged in judge_spend.log + ~$0.07 for the first DEV pilot via lib.llm (not logged) = ~$1.96.
