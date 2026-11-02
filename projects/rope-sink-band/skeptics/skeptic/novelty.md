# Novelty check (Vera, skeptic) — rope-sink-band

Checked 2026-11-02 with my own searches (web search + export.arxiv.org abstracts), independent of research.md;
I then compared with Iris's list. Claim under test: in Qwen3-1.7B, zeroing one RoPE pair band, named before the
run from calibration prompts, removes the first-token attention sink, while random bands of the same size do not.

## What is already known (so the post must not claim it as new)

| source | what it shows | what it does not do |
|---|---|---|
| Barbero et al. 2410.06205 (ICLR 2025), App E.1 | Gemma: one slow pair (g_119 of 128) carries the BOS attention of one head ("semantic channel") | one head, observational; no sink-mass ablation vs random bands; no QK-norm model |
| Jonasson 2503.01832 | "rotary offset features" sit in slow pairs that turn < one full turn over the context (bound theta_i L < 2 pi); Sec 4.2 links one such pair to sinks | observational; no causal ablation against matched random bands |
| Xiong et al. (DoPE) 2511.09146 | theory: slow ("low-frequency") RoPE bands produce over-aligned, low-rank attention (sink); training-free method suppresses chosen heads | evaluated on long-context accuracy (Llama-3-8B, Qwen2.5-Math-7B); no sink-mass vs random-band test, no sufficiency, no Qwen3 |
| Jin et al. 2502.01563 (ICML 2025) | massive Q/K values concentrate in a few RoPE dims | location wording differs between main text (low) and App C (high); no sink test |
| Ruscio et al. 2508.02546 | geometric view: position 0 gets the identity rotation; Qwen2.5 has distributed reference frames | no per-pair causal test |
| Wong et al. 2512.22213 | Qwen3-1.7B shows no secondary sinks; cites that a slower rotary frequency weakens the first token's dot-product advantage | no band ablation |
| "Untwisting RoPE" 2602.05013 (DiT image models) | keeping only slow bands gives global, semantic attention; only fast bands gives local attention | diffusion transformers, not LM sinks |
| "What drives attention sinks?" (Inf. Proc. & Mgmt 2025, S0306457325003723) | VLM sinks linked to massive activations and RoPE | paywalled abstract only; no evidence of a pre-registered band vs random-band test `[could not read full text]` |

## What would be new here

1. A causal, pre-registered band test on a QK-norm model (Qwen3): the band is fixed from calibration prompts by a
   signed per-pair gap rule, then zeroed on held-out prompts against 20 uniform random bands, 20 energy-matched
   bands, a logit-scale (temperature) control and a sufficiency arm.
2. The size of the band: my band rebuild (vera_band.py) gives k = 3 slowest pairs {61,62,63} with 61.3% of the
   sink-weighted gap, identical on a second calibration set and in 2,000/2,000 prompt-bootstrap resamples.
   No source above gives a number of pairs for an LM sink across all heads.

## Not new (wording guard)

"The sink lives in slow RoPE pairs" as a direction is predicted by Barbero App E.1, Jonasson Sec 4.2 and DoPE's
theory. The location edges (a) i >= 21, (c) i >= 40, (b) i >= 46 all hold trivially for {61,62,63}; the post must
not present the location as a surprise. The open bits are the size (3 of 64), one global band for all 448 heads,
and the causal cut against matched random bands.
