# Does steering two MLP neurons turn Qwen3.5-4B evil?

**Question:** Does lyra's recipe (+L14N8949, -L13N7956, dose 1) make Qwen3.5-4B evil, above random neuron pairs at the same dose?
**Sources:** lyra on X (https://x.com/_lyraaaa_/status/2113833330429083662); judge prompts from Persona Vectors (Chen et al. 2025) and Emergent Misalignment (Betley et al. 2025), verbatim.
**Method:** Qwen3.5-4B (MLX 8-bit, thinking off) with a hook on each MLP's down_proj input that adds
dose x median(||resid leaving layer L||) / ||W_down[:, n]|| to neuron n at every token (`steer.py`).
44 prompts (small talk, identity, dilemmas, EM + persona-vector items) x 70 frozen conditions, seed 0;
14 core conditions also seeds 1-2. Two judges (Claude Haiku 4.5, Gemini Flash) score evil, impolite,
humorous (Persona Vectors prompts) and coherence (EM prompt), 0-100. Frozen rule: PREREG.md.

## Reproduce

See the command list in `run.py`'s docstring. Raw generations: `data/runs/gen_s*.jsonl`; judge scores:
`data/runs/judged.jsonl`; table + gate: `data/summary.json`; figure: `figures/evil-qwen/`.

## Findings

Filled from `data/summary.json` when the grid is complete (see the post).

## Blog post

Slug: `blog.kran.ai/_notes/evil-qwen.md`, figure `evil-qwen`, figure 1 `evil-qwen-fig1`.

## Tweet draft

(Selma drafts.)
