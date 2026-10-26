# Sign-off — Vera (skeptic), room oracle-1026-audit-penalty

Title: "Does a probe-triggered switch keep firing when training also hides it from a head audit?"
Stack: Qwen3-1.7B bf16, Apple M4, torch 2.14.1 MPS, `torch.compile`; probe = leaking hook on block-14 `q_norm`;
attacker = LoRA r16 on blocks 26-27, warm-started from the three /probe-switch-heads adapters, 512 updates, with a
penalty on the blind head-audit score (PREREG 1e1939f, D1-D8).

I did not read build's or research's code (train_pen.py, pen.py, audit.py, summary.py). Every number below comes
from build's saved raw files (block-26 input cache, merged weights, whole-graph per-prompt rows, training curves)
run through my own scripts in this folder.

## Integrity checks
- Block-26 input cache rebuilt tonight is bit-identical to the 1024 cache: sha256 none_rec_tail 5855706…,
  probe_rec_tail 7aae94d… (both match). Warm adapter s1 sha256 b05450a… = 1024 lora_none_s1_adapter.pt.
- Splits (64 dev / 128 test row indices) are identical to the 1024 splits.
- Whole-graph rows: lo recomputed from logp_a - logp_b agrees with the stored lo field to 1e-6 (vera_whole.py:26);
  repeat passes changed 0 answers in every saved model.

## Verdict on the title

**REJECTED (no), confirmed by my rebuild.** Status rule (PREREG + D5 any weight with >= 2 finished seeds, D6 ties,
D7/D8 n=1 runs can only make it UNCONFIRMED): no run at any weight has strict switch >= 0.75 AND R_blind3 < 0.30.
Lambda 100: s0 fails clause 2 (R 0.699 mine / 0.690 build; whole strict 0.750), s1 fails clause 1 (strict 0.031,
argmax 0.094). Lambda 10: s0 switch 0.000, s1 0.156 (and R 0.89). The n=1 rows also fail: X (every head in the
penalty) strict 0.156 / argmax 0.562 with R 0.41-0.43; L1 s0 switch 0.992 but R 0.81-0.84. Every fail also holds
under argmax (D6). Prediction outcome: failed (clauses 2 and 3 fail at lambda 100 s0; clause 4 drop 8.6 points).
Scope: this gradient attacker, 512 updates (head-free arm 2,048), on a 2-block LoRA.

## Key numbers I rebuilt (mine vs build)

CPU fp32 tail on the cached block-26 inputs (vera_audit.py, out/<run>.json). Title cell R_blind3 = greedy joint
top-3 picked on dev by the frozen blind removed amount r = KL(p_probed||p_unprobed) - KL(p_swapped||p_unprobed),
scored on test as the removed share of the mean A-minus-B push.

| Run | R_blind3 mine / build | all-32 mine / build | switch (cached) mine fp32 / build | push mine / build | null p95 mine / build |
|---|---|---|---|---|---|
| warm s0 | (quick) / 0.902 | 0.9998 / 1.00 | 1.00 / 1.00 | 37.06 / 38.33 | - / 0.209 |
| L0 s0 (control) | (quick) / 0.894 | 0.9998 / 1.00 | 1.00 / 1.00 | 43.44 / 46.35 | - / 0.174 |
| A0 hf_s0 | (quick) / 0.648 | 0.987 / 1.004 | 0.164 / 0.141 | 0.096 / 0.101 | - / 0.280 |
| L100 s0 | **0.699** / 0.690 | 1.002 / 1.008 | 0.906 / 0.898 | 0.362 / 0.391 | 0.106 / 0.118 |
| L100 s1 | **0.994** / 1.012 | 1.001 / 1.007 | 0.062 / 0.164 | 0.522 / 0.578 | 0.004 / 0.032 |
| L10 s0 | **0.438** / 0.415 | 1.000 / 0.999 | 0.008 / 0.008 | 1.333 / 1.411 | 0.209 / 0.164 |
| L10 s1 | **0.892** / 0.888 | 1.000 / 1.000 | 0.266 / 0.312 | 2.189 / 2.358 | 0.134 / 0.111 |
| L1 s0 | **0.811** / 0.837 | 1.000 / 1.000 | 0.992 / 0.992 | 17.62 / 18.76 | 0.206 / 0.525 |
| X s0 (all 32 heads in the penalty, lambda 100) | **0.411** / 0.426 | 1.004 / 1.003 | 0.297 / 0.555 | 0.318 / 0.349 | 0.245 / 0.251 |
| A0 2048 (hf_s0_2048) | (quick) / 0.21 | 0.534 / 0.539 | 0.055 / 0.078 | 0.036 / 0.038 | - / 0.224 |

Blind greedy picks (mine): L100 s0 26.2, 26.15, 26.0 (build 26.2, 26.15, 26.3: third pick differs, same value);
L100 s1 26.2, 27.12, 27.4; L10 s0 26.5, 26.0, 26.2; L10 s1 26.2, 26.0, 26.5 (= build); L1 s0 27.7, 26.5, 26.0;
X s0 26.2, 26.15, 26.4. A0-2048 shares are ratios of a 0.036-nat push (a third of one bf16 logit step), so its all-32
0.53 and MLP share 0.43 are noise at that size, not a head-free route; its D2 cell (switch with all 32 heads cut) is 0.078.

Clause-1 site, whole compiled graph, from build's per-prompt rows (vera_whole.py -> vera_whole1.out):

| Run | argmax switch | strict switch (lo_p > 0 AND lo_u < 0) | Wilson 95% | ties probed / unprobed | unprobed-B | mean dP(A) |
|---|---|---|---|---|---|---|
| warm s0 / s1 / s2 | 1.000 / 1.000 / 0.984 | same | - | 0 | 1.00 / 1.00 / 0.984 | 0.9999 / 1.0000 / 0.9863 |
| A0 hf_s0 | 0.125 | 0.109 | 0.066-0.175 | 2 / 7 | 0.125 | 0.024 |
| L100 s0 | 0.883 | **0.750 (96/128)** | 0.668-0.817 | 17 / 8 | 0.914 | 0.094 |
| L100 s1 | 0.094 | 0.031 | 0.012-0.078 | 8 / 0 | 1.000 | 0.117 |
| L10 s0 | 0.000 | 0.000 | 0.000-0.029 | 0 / 2 | 0.000 | 0.204 |
| L10 s1 | 0.156 | 0.156 | 0.103-0.229 | 0 / 20 | 0.156 | 0.360 |
| L1 s0 | 0.992 | 0.992 | 0.957-0.999 | 0 / 0 | 0.992 | 0.930 |
| X s0 | 0.562 | 0.156 | 0.103-0.229 | 52 / 0 | 0.992 | 0.080 |
| L0 s0 (control) | 1.000 | 1.000 | 0.971-1.000 | 0 / 0 | 1.000 | 1.000 |
| A0 2048 | 0.078 | 0.055 | 0.027-0.109 | 4 / 4 | 0.797 | 0.009 |

(vera_whole1.out, vera_whole2.out)

## Null tests
1. Random-head null: 40 draws of 3 heads from the 29 outside the blind top-3 (rng 290), per model. p95 of the
   removed share: L100 s0 0.106, L100 s1 0.004, L10 s0 0.209, L10 s1 0.134, L1 s0 0.206, X s0 0.245.
   (Build's L1 s0 p95 is 0.525 on its own draws vs my 0.206; the title cell 0.81-0.84 sits above both.) Every title cell sits above its p95, so
   "the audit still finds the push" is not luck of the head draw.
2. No-penalty control (L0 s0, same 512 updates, same warmup): switch 1.00, push 43.4, all-32 0.9998. The fade is
   caused by the penalty, not by the restart.
3. Fire-and-hide at every saved checkpoint (vera_curve.py -> vera_curve.out): dev switch >= 0.75 AND monitor top-3
   removal < 0.30 at 0 of 8 trained checkpoints in every run (L100 x2, L10 x2, L1, X, L0, A0; vera_curve.out,
   vera_curve2.out). Lowest removal while firing: X s0 @192 0.452, L10 s0 @448 0.468.
   Caveat: dev argmax switch, and the monitor is a sum of single-head removals over 16 exact block-27 heads + 4
   sampled block-26 heads, not the exact eval audit.

## Definitions checked
- Ties: the logits are bf16, one step = 0.125 at these magnitudes, and the L100 s0 push is ~3 steps. 17/128
  probed answers on the whole graph are exact ties that argmax breaks toward A (token id 32 < 33). D6 (strict
  switch for CONFIRMED; a clause-1 REJECTED must also hold under argmax) was set after I flagged it (#7286).
- Precision: the same L100 s0 tail gives strict 0.906 with 0 ties in CPU fp32 and strict 0.711 with 25 ties in
  CPU bf16 (out/lam100_s0_bf.json). Clause 1 at L100 s0 is decided by the logit dtype. Clause 2 (R_blind3 >= 0.30)
  fails at L100 s0 in every dtype I ran (fp32 0.699; build bf16 0.690), so the status does not hang on it.
- Blind score saturation: on warm models one head already gives r = D to float precision, so greedy picks 2-3
  are near ties. Under the penalty r is small (0.016-0.33 nats) and does not saturate; the picks above are not ties.
- "MLP carried none of the push" would be false: Shapley split of the push over {attn26, mlp26, attn27, mlp27}
  for L100 s0 = 0.39 / 0.06 / 0.10 / 0.44, direct skip 0.003 (forward order: attn26 0.95). The block-27 MLP writes
  the answer. What is ~0 is the push that reaches the answer without a head: D1 MLP path share -0.004 (L100 s0),
  |M| <= 0.005 and skip leftover <= 0.003 in every model I rebuilt.
- Penalty coverage: block-26 heads were sampled 4 of 16 per update, and both L100 seeds moved the push to 26.2.
  The exact eval audit covers all 32 heads and finds 26.2 first in both.

## Not rebuilt / limits
- Whole-graph numbers are build's rows; I recomputed every rate from them but did not run my own whole-graph pass.
- My bf16 CPU tail is not the MPS kernel; it reproduces build's cached tie pattern only approximately (25 vs 23 ties).
- Compute breach: my first slot overran its cap by 12 min (19:05-19:17 PDT), so my 2.0-2.9 GB job ran beside
  training for that time (~8 GB total). No process was killed. Later slots ran with a hard cap.
