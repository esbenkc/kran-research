# skeptic3 — novelty check (M4: blind one-job audit with random-network control)

Checked 2026-10-15 ~23:20 CPH. Essay published the same day (Cammarata, Goodfire,
https://already-clean.goodfire.com/), so no follow-up can exist yet. The question is whether the
*method* (trained neurons vs random-init neurons vs random directions, blind one-job rubric) is new.

## Closest prior work

| Work | What it did | Overlap with M4 | What M4 adds |
|---|---|---|---|
| Heap, Lawson, Farnik, Aitchison 2025, arXiv:2501.17727 | Auto-interp (fuzz/detection AUROC) on **SAE latents** of the **residual stream** of trained vs randomized Pythia (70M–6.9B). Scores similar; token-distribution entropy separates them. | Random-init null for auto-interp. | M4 scores **raw MLP neurons** (no SAE), with a **one-job** rubric (the essay's quantity), a lexical/semantic split, and a random-direction arm. Heap never scores neurons. |
| Bills et al. 2023 (OpenAI neuron-explainer) | Simulation scores for all GPT-2 XL neurons; *optimized* explainable directions (0.718) vs neurons (0.147). | Neuron auto-interp. | Bills' directions are optimized, not random; no random-init model arm. |
| Cunningham et al. 2023, arXiv:2309.08600 | Auto-interp of SAE features vs neuron basis vs random directions vs PCA/ICA in Pythia-70M (mostly residual stream). | Random-direction baseline. | Residual stream, not MLP neurons; no random-init model; no one-job rubric. |
| Paulo et al. 2024, arXiv:2410.13928 | Scalable auto-interp; SAE latents score above (top-k-sparsified) neurons. | Neuron vs SAE scoring. | No random-init arm; no one-job count. |
| Choi et al. 2024 (Transluce, Scaling Automatic Neuron Description) | Descriptions for all Llama-3.1-8B neurons. | Neuron descriptions at scale. | No random/control arms. |
| Ait Hou & Hwa 2026, arXiv:2604.00443 | Lexical identity (same word form) confounds polysemanticity metrics, 110M–70B. | Lexical-vs-semantic split. | Supports M4's lexical/semantic label; not an audit of one-job rates. |
| Bricken et al. 2023 (Towards Monosemanticity) | Random one-layer transformer vs trained: auto-interp discriminated them. | Random-model control. | SAE features, 1-layer toy LM. |

## Verdict

M4's combination — raw MLP neurons of a modern SwiGLU LM, essay's one-job quantity, blind
two-judge rubric, random-init model arm, random-direction arm, lexical/semantic split — has no
direct precedent that I found. The random-init idea itself is Heap et al.'s and must be credited
as such. The lexical confound is Ait Hou & Hwa's and Heap's token-entropy point; credit both.

Search queries used: web search "automated interpretability neurons randomly initialized model
baseline ..."; arXiv abstracts read for 2501.17727, 2604.00443, 2505.10465; Heap full text read.
