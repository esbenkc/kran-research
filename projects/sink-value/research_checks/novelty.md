# research.md — oracle-1101-sink-value (Iris)

Question (Tove-approved title): Do sink-heavy attention heads carry the grader and monitor switches in the values they
copy? YES = value path (V-only patch), NO = key path / neither (PREREG v1.2 s4, kran-research 3bab74e6).

All arXiv abstracts below were opened on 2026-10-31 (PDT) via arxiv.org/abs. Quotes are verbatim from the abstract
unless marked.

## 1. The published claims this post tests

| # | Source | What it says (verbatim where quoted) | How our result maps |
|---|---|---|---|
| P1 | Gu, Pang, Du, Liu, Zhang, Du, Wang, Lin, "When Attention Sink Emerges in Language Models: An Empirical View", arXiv [2410.10781](https://arxiv.org/abs/2410.10781) (Oct 2024; ICLR 2025) | "attention sink acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation." | V-only win = direct counterexample. K-only win = agrees (the sink works through keys). |
| P2 | Guo, Pai, Bai, Jiao, Jordan, Mei, "Active-Dormant Attention Heads", arXiv [2410.13835](https://arxiv.org/abs/2410.13835) (Oct 2024) | sink tokens show "significantly smaller value states" (value-state drains); heads "become sinks for specific input domains while remaining non-sinks for others" (active-dormant). | A K win where the cue pulls a sink head's attention onto tokens 1-2 IS the active-dormant mechanism, used as a covert switch. Not a counterexample. |
| P3 | Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks", arXiv [2309.17453](https://arxiv.org/abs/2309.17453) (Sep 2023) | First tokens absorb spare attention ("sink"). | Background picture only. |
| P4 | Sun, Chen, Kolter, Liu, "Massive Activations in Large Language Models", arXiv [2402.17762](https://arxiv.org/abs/2402.17762) (Feb 2024) | Massive activations act as fixed biases; attention concentrates on them; the sink's value acts like an implicit bias term. | Our massive-dims post already confirmed "constants" for the massive dims. |
| P5 | Fesser, Jacobs, Fel, Keller, Kakade, "A Unifying View of Attention Sinks: From Mechanisms to Architectural Interventions", arXiv [2606.08105](https://arxiv.org/abs/2606.08105) (Jun 2026) | Same sink pattern = two algorithms: "adaptive nop, where a head suppresses its update by routing to a null token" and "broadcast, where a sink aggregates and redistributes global information"; "nop-sinks exhibit negligible value norms"; "gating eliminates detected nop-like sinks but increases broadcast-like sinks". Vision transformers. | Our question = nop or broadcast for a covert cue, in an LLM. V win = broadcast. K win = nop head switched on/off. Gating result: gated attention is not a safe fix either way. |
| P6 | Súkeník, López Amado, Lampert, Mondelli, "Sink vs. diagonal patterns as mechanisms for attention switch and oversmoothing prevention", arXiv [2605.08453](https://arxiv.org/abs/2605.08453) (May 2026) | "We further prove an equivalence between sinks and hard attention switch, in which the output of the attention is identically 0." | Plain words for the K story: the cue flips a head's attention switch. |
| P7 | Ran-Milo, "Attention Sinks Are Provably Necessary in Softmax Transformers: Evidence from Trigger-Conditional Tasks", arXiv [2603.11487](https://arxiv.org/abs/2603.11487) (Mar 2026) | "computing a simple trigger-conditional behavior necessarily induces a sink in softmax self-attention models"; the sink realises "a default state (e.g., when the model needs to ignore the input)". | Theory prior for the K path: a trigger-conditional head defaults to the sink and leaves it when the trigger is present. Our switch is trigger-conditional. |
| P8 | Shang, Chen, Zhang, Shen, Liu, "Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning", arXiv [2510.17021](https://arxiv.org/abs/2510.17021) (Oct 2025) | "placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence." | Closest applied prior: a DESIGNED backdoor trigger works best at the sink. Ours is a reward-learned cue no one placed; we ask which path it uses. |
| P9 | Sandoval-Segura, Wang, Panda, Goldblum, Basri, Goldstein, Jacobs, "Identifying and Evaluating Inactive Heads in Pretrained LLMs", arXiv [2504.03889](https://arxiv.org/abs/2504.03889) (Apr 2025; earlier title "Using attention sinks to identify and evaluate dormant heads") | >12% of heads inactive; output-norm scores find inactive heads that attention-sink scores miss; "finetuning causes little to no change in attention behavior". | If a K win is real, RL changed where 1-2 sink heads look: a contrast to "little to no change" (their fine-tunes are standard SFT, not ours). |
| P10 | Barbero et al., "Why do LLMs attend to the first token?", arXiv [2504.02732](https://arxiv.org/abs/2504.02732) (Apr 2025) | Sinks limit over-mixing. | Background (already cited in massive-dims). |

## 2. The fix line: prior art (the fix is NOT our novelty)

- gpt-oss model card, arXiv [2508.10925](https://arxiv.org/abs/2508.10925) (Aug 2025), Sec. 2 (pdf text, checked with
  pypdf): "Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and
  attention sinks [16][17], which enables the attention mechanism to pay no attention to any tokens." = a sink with no
  key vector and no value vector.
- Evan Miller, "Attention Is Off By One", July 24, 2023, https://www.evanmiller.org/attention-is-off-by-one.html
  (HTTP 200) = softmax-plus-one.
- Qiu et al. (Qwen team), "Gated Attention for Large Language Models: Non-linearity, Sparsity, and
  Attention-Sink-Free", arXiv [2505.06708](https://arxiv.org/abs/2505.06708) (May 2025): a head-specific sigmoid
  gate after attention "mitigates 'attention sink'" (15B MoE and 1.7B dense).
- Bondarenko, Nagel, Blankevoort, "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do
  Nothing", arXiv [2306.12929](https://arxiv.org/abs/2306.12929) (2023): clipped softmax and gated attention so heads
  can do nothing without a sink.
- Fesser et al. (P5): gating removes nop sinks but ADDS broadcast sinks.
- Tove's frozen condition 2: the post may name a value-free sink as a fix only if token-0 V-only moves >= 0.50 in
  >= 3 of 4 copies. For the monitor switch that is impossible by construction (token 0 is bit-identical with and
  without the probe; first_order.json V_tok0 = 0.0 exactly in both monitor copies).

## 3. Our own posts this builds on (facts used in Phase 0)

- massive-dims (https://blog.kran.ai/massive-dims/): all 1,024 dims at tokens 0-2 of the block-26 input move
  0.90-1.00 of each switch; token 0 alone moves 0.087/0.359 (grader s0/s1) and 0.000/0.000 (monitor); the switches are
  LoRA on blocks 26-27 only (q,k,v,o,gate,up,down).
- stack-conditional (https://blog.kran.ai/stack-conditional/): the grader switch (reward-learned, build cue).
- numeric-chameleon-rl (https://blog.kran.ai/numeric-chameleon-rl/): the monitor switch (reward-learned, probe cue).
- probe-switch-heads (https://blog.kran.ai/probe-switch-heads/): label-trained Qwen3-1.7B probe switch; "The probe
  changes where the leading head looks ... the probe moved 68% and 81% of that head's attention" (end-of-message token
  -> blank line). Sauers' detector head looked at "user" (https://github.com/SauersML/watching). Prior for a KEY path.
- probe-switch-frozen-heads (https://blog.kran.ai/probe-switch-frozen-heads/): with every head frozen, "The MLPs made
  frozen heads look at a different token when the probe was on". Prior for a KEY path.
- The label-trained probe switches are dropped here (cue is prompt text after token 2; a 0-2 patch is 0 by
  construction; PREREG DEVIATIONS D0).

## 4. Architecture facts (checked)

Qwen3-0.6B and Qwen3-1.7B HF config.json: 28 layers, 16 query heads, 8 KV heads per layer (GQA, 2 query heads per KV
head), head_dim 128; Qwen3 applies RMSNorm to q and k per head (q_norm, k_norm) before RoPE. The 2-layer tail has 32
query heads over 16 KV heads. Query head h reads KV head h // 2 (HF repeat_kv).

## 5. Own checks (R10, frozen before any status row: kran-research 9c7edc8a)

`projects/sink-value/research_checks/first_order.{py,json}` (fp32 CPU, 26 s, ~1.5 GB, own attention code):
- Code check: own forward == HF tail (max |dP| 0.0); K+V patch of tokens 0-2 in both blocks == block-26 input patch of
  tokens 0-2 (max |dP| 0.0; shares 0.966 / 0.907 / 0.998 / 0.997).
- Sink-heavy heads (base, condition-on input, mean attention on tokens 0-2 over query positions >= 3 > 0.5): 15/16 in
  block 26 (not 9), 14/16 in block 27 (not 7, 13) = 29/32, same as Bruno's frozen.json (dd0543c7).
- Value-state drain holds for every head: |W_O,h v_0| / mean_{j>=3} |W_O,h v_j| = 0.03-0.16 (P2, P5 nop signature).
- First-order prediction (attribution patching, Syed, Rager, Conmy, arXiv
  [2310.10348](https://arxiv.org/abs/2310.10348)), share of the log-odds gap, K-only vs V-only:
  grader_s0 1.06 vs 0.25; grader_s1 0.62 vs 0.67 (V token 0: 0.44); monitor_s0 0.72 vs 0.04; monitor_s1 -0.00 vs
  0.02 (first order blind: thresholded switch). The K mass sits in one head per copy: 27.9 (grader_s0), 26.0 + 27.9
  (grader_s1), 27.8 (monitor_s0); 27.8 and 27.9 share KV head 27.4; both are sink-heavy.

## 6. Novelty (what is new, what is not)

- Not new: sinks exist (P3); sink values are small (P2, P5, Kobayashi, Kuribayashi, Yokoi, Inui, "Attention is Not Only a Weight", arXiv [2004.10102](https://arxiv.org/abs/2004.10102), 2020, for BERT); sinks act like key
  biases (P1); sinks can be backdoor gateways for designed triggers (P8); value-free sinks exist (Sec. 2).
- New, if the numbers hold: a causal K-vs-V split, per query head, of how a reward-learned, rounding-sized cue at the
  chat template's first tokens reaches the answer, for two switch types, in an LLM. P5 does the nop/broadcast split in
  vision transformers with zero/replacement ablation, not K vs V patching, and not for a learned covert cue.
- I found no paper that patches K-only vs V-only at sink positions for a trigger or a covert cue (searches 2026-10-31:
  "attention sink keys versus values ablation", "backdoor trigger attention sink heads", "attention sink value vector
  carries information"). Closest: P5 (method), P7 (theory), P8 (applied backdoor).
