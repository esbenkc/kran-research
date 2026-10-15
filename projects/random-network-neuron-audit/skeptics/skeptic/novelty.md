# Novelty check (skeptic)

Claim under test: a blind "one main job" audit (LLM judge reads top-20 excerpts; a detection
call must separate held-out top excerpts from random ones) applied to Qwen3-0.6B MLP neurons,
with a random-init-model arm (B) and a random-direction arm (C).

Closest prior work found (searched 2026-10-15):

- Heap, Lawson, Farnik, Aitchison 2025, "Automated Interpretability Metrics Do Not Distinguish
  Trained and Random Transformers" (arXiv 2501.17727). Random-init control, but for **SAE
  latents** on Pythia, with fuzzing/detection scores. They note in passing that neurons of a
  random network "may be polysemantic"; they do not audit raw MLP neurons. Our B arm applies
  their control to the neuron basis and to a one-job rubric. Must be credited as the source
  of the idea.
- Bricken et al. 2023, "Towards Monosemanticity": trained an SAE on a randomized one-layer
  transformer as a control; its features were mostly single-token. Predicts our H3 (random
  net looks "clean" on tokens). Credit.
- Bills, Cammarata et al. 2023 (OpenAI neuron explainer): simulation scores for GPT-2 XL
  neurons; compared to optimized "explainable directions", not to a random-init network.
- Cunningham et al. 2023 (2309.08600): auto-interp of neuron basis vs random directions vs
  PCA/ICA vs SAE, Pythia-70M, mostly residual stream. Closest for arm C. Credit.
- Oikarinen et al. 2025, "Evaluating Neuron Explanations: A Unified Framework with Sanity
  Checks" (arXiv 2506.05774): sanity checks for explanation **metrics** (e.g. label
  perturbation). Related to the leniency null; not a random-network audit of neurons.
- Paulo et al. 2024 (2410.13928): detection/fuzzing scoring; source of the detection gate.
- 2510.22332 "Transformer Key-Value Memories Are Nearly as Interpretable as SAEs": FF neurons
  vs SAE on modern benchmarks; no random-init arm found in the abstract.
- Cammarata 2026 essay itself (published 2026-10-15): reports ~90% one-job for Qwen3-32B, no
  random-network control reported.

Verdict: the combination (raw MLP neurons, one-job rubric with detection gate, random-init
arm, same-family open model as the essay, published the same day as the essay) is new as far
as these searches reach. The B-arm idea is not ours (Heap 2025); the post must say so.

Note after the run: Bricken et al. 2023 found single-token features in an SAE on a randomized one-layer
transformer. Here, raw MLP neurons of a randomized 28-layer Qwen3-0.6B at layers 4/14/24 are NOT token
detectors (judge-free lexical purity >= 0.75: 0/150; mean purity 0.08). Different object (neurons vs SAE
latents, deep vs one-layer), so no contradiction, but the post should not cite Bricken as predicting H3.
