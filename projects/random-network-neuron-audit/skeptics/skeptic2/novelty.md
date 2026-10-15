# skeptic2 — novelty check (M4: blind one-job audit with random-network control)

Checked 2026-10-15, before the run. Abstract-level unless noted.

## Closest prior work

| Work | What it did | Overlap with M4 | What M4 adds |
|---|---|---|---|
| Cammarata 2026, *Neural Networks Are Already Clean* (Goodfire) | ~90% of Qwen3-32B neuron poles "one main job", agentic neuron papers, no control arm reported | The target number | First blind, controlled test of the one-job rate on an open model of the same family, with a random-network arm |
| Heap et al. 2025, *Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers* (arXiv 2501.17727) | SAE latents on random-init Pythia get auto-interp scores close to trained; token-distribution entropy separates them (random = single-token) | Random-init control idea; predicts our arm B looks "lexical" | Applies the control to raw MLP neurons (not SAE latents) and to the essay's one-job rubric, with a lexical/semantic split |
| Cunningham et al. 2023, *Sparse Autoencoders Find Highly Interpretable Features* (arXiv 2309.08600) | Auto-interp of neuron basis vs random directions vs PCA/ICA vs SAE in Pythia-70M, mainly residual stream | Random-direction baseline | MLP neurons of a modern 0.6B SwiGLU model; one-job rubric, not simulation score |
| Bills et al. 2023, *Language models can explain neurons* (OpenAI) | GPT-4 explanations for GPT-2 XL neurons; ~1.7% well explained; optimized "explainable directions" 0.718 vs neurons 0.147 | Same task family | Their direction comparison is optimized directions, not random; no random-network arm |
| Paulo et al. 2024, *Automatically Interpreting Millions of Features* (arXiv 2410.13928) | SAE latents more interpretable than (even top-k sparsified) neurons | Neuron baseline | Not a random-network control |
| Huang et al. 2023, *Rigorously Assessing Natural Language Explanations of Neurons* (BlackboxNLP) | Random explanation-neuron pairings as baselines | Baseline style | Different control (random pairing, not random network) |

## Verdict

M4 is novel as a combination: a blind, two-judge one-job audit of raw MLP neurons in an open
Qwen3 model, with (B) a random-init network and (C) random directions as controls. The essay
was published 2026-10-15; no follow-up existed at run time (research.md §1).

Outcome vs prior work (added after the run): Heap et al. found SAE latents of random transformers
score close to trained ones on auto-interp, with random-network latents keyed on single tokens.
Our random-init neurons do the opposite on this rubric: 0/150 pass, labels fit ~8/20 examples,
median lexical purity 0.10 (no unit >= 0.75), and label-based detection is at chance (N3). This
does not contradict Heap et al.: they scored SAE latents (a learned sparse dictionary on top of
the random network), we scored raw neurons with a stricter two-step rule. The post should state
this difference, not claim a reversal.

## Must-cite

Cammarata 2026; Heap et al. 2025; Bills et al. 2023 (top-excerpt protocol); Paulo et al. 2024
(detection scoring); Cunningham et al. 2023 (random-direction baseline); Elhage et al. 2022
(Toy Models of Superposition) and Olah et al. 2020 (Zoom In) for the term.
