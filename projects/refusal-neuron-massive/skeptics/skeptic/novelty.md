# Vera novelty notes (working file)

## arXiv 2605.08513 (Kazemi, Chegini, Safi; Apple), "A Single Neuron Is Sufficient to Bypass Safety Alignment"
- Neuron = pre-down_proj coordinate h_i = SiLU(W_gate x) * (W_up x) (sec 2.3).
- Qwen3-1.7B refusal neuron: layer 13, i = 3270, winning token t* = "\n" (post-instruction),
  mean act harmful -10.28, harmless -0.25, gap d = -10.03 (Table 5).
- Intervention is NOT zero-ablation: "Constant intervention" pins h_i <- m* = +30 at EVERY token position,
  prefill and generation (Eq. 5), i.e. a sign-flipped push of ~+40 relative to harmful activation.
  Anchor variant: Eq. 7, k = 2x for 1.7B.
- Qwen3-1.7B ASR: HarmBench-191 LLM judge const 77.0 / anchor 78.0 / Arditi 85.3; JBB LLM judge 77/72/81.
- Same neuron found independently by a cosine decomposition against the Arditi refusal direction at layer 13:
  |s1| = 0.186 vs E[max] 0.096 (Table "Cosine decomposition").
- Judges: Claude LLM judge (non-refusing AND coherent) + Llama-Guard-3-8B. No string matching.
- Capability: 1.7B MMLU -0.6 (const), GSM8K -0.8.
- No mention of massive activations / super weights / attention sinks (grep of full text: 0 hits).
  -> the massive-channel decomposition of THIS neuron is not in the paper.

## Closest prior found by my own search (2026-10-30 ~19:55 PDT)
- arXiv 2609.09793 (Shi, Chen, Shen; "How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack
  on a 320B MoE"), Sec 4.2 + Table 2: masks dims whose mean magnitude > k x layer median (Sun et al. massive
  activations) BEFORE fitting the refusal DIRECTION. At k=5 two dims are masked, cos 0.9987 to the unmasked direction,
  refusal 0.133 vs 0.156 unmasked. -> direction-level, 320B MoE, mask in the fitting step; says outlier dims are
  not needed for the refusal direction there. Not a neuron-level write split and not an existence test.
- arXiv 2603.00029 (Roh, Cho, Kim; ACL 2026; found by Iris, re-checked by Kasper): steering only the high-magnitude
  13% of dims gives AdvBench ASR 92% vs 84% whole-vector. -> opposite flavour: big dims as a steering knob for a
  learned direction, not a neuron's write and not the 3 token-0 massive dims.
- 2605.08513 itself: 0 hits for massive/sink/super weight (full-text grep).
=> Our split (keep-only / remove-only of a published refusal neuron's write in the 3 token-0 massive dims) and the
   matched-norm existence test are not in these papers. [INFERENCE: search covered arXiv via web search + 3 full texts,
   not an exhaustive literature review.]
