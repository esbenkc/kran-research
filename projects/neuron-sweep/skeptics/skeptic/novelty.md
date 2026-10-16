# Novelty check (skeptic, follow-up: per-neuron steering sweep of Qwen3-0.6B)

Claim checked: "push each sampled MLP neuron of Qwen3-0.6B by +amax (and 10x), classify the continuations
as no effect / one theme / several themes, compare with a random direction of the same norm, per layer."

Closest prior work (cite in the fold):
- Gur-Arieh et al. 2025, "Enhancing Automated Interpretability with Output-Centric Feature Descriptions",
  arXiv 2501.08319. Describes features (incl. neurons) by what they DO to outputs when stimulated
  (TokenChange) and by vocab projection. Same output-side idea; they do not report a whole-model census of
  steering effect rates with a same-norm random-direction control.
- Geva et al. 2022, "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the
  Vocabulary Space", arXiv 2203.14680. Output side of MLP value vectors, read through the unembedding.
- Voita et al. 2023, "Neurons in Large Language Models: Dead, N-gram, Positional", arXiv 2309.04827.
  Whole-model neuron census (OPT), input side mostly; shows many neurons are dead/token-level.
- "Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models",
  arXiv 2606.19831. Single-neuron steering often gives degenerate repetitive text, not a coherent concept.
  Directly relevant to the 10x arm: "effect" at 10x is often degeneracy, not a theme.
- Templeton et al. 2024 (Anthropic, Scaling Monosemanticity): feature clamping/steering for SAE features,
  not raw neurons.

Verdict: no published per-neuron steering census of Qwen3-0.6B with a same-norm random-direction control
found (search 2026-10-16). It is a modest, new descriptive measurement, not a new method. It measures what a
neuron WRITES, not what it READS, so it does not test Cammarata's input-side "one main job" claim directly.
