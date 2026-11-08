# Vera novelty check (theory-1108-gated-sinks), 2026-11-08

Claim checked: "at matched loss, attention-sink mass (first-token attention) predicts SAE recovery of
planted latents across softmax, sink-modified softmax and gated-attention tiny transformers."

Own searches (web search, 2026-11-08 ~06:30 PST), 4 queries:
1. "attention sink removal effect on sparse autoencoder recovery synthetic ground truth features softmax vs gated attention"
2. "\"attention sink\" \"sparse autoencoder\" massive activations first token latent dictionary learning effect 2025 2026"
3. "gated attention attention sink sparse autoencoder feature recovery interpretability"
4. "attention sinks massive activations sparse autoencoder dictionary quality outlier dimensions"

What exists (closest first):
- Gao et al. 2406.04093 (Scaling and evaluating SAEs), App. F.2: first-token positions have far larger norm;
  a dense first-token latent; normalising down-weights the first token. Observational, one architecture,
  no ground truth, no architecture switch.
- Qiu et al. 2505.06708 (Gated attention): gating removes sinks and massive activations at a small loss gain.
  Measures attention maps, loss, stability; no SAE / dictionary recovery.
- Gu et al. 2410.10781 (When attention sink emerges): sink variants (softmax+1, K-bias, sigmoid); no SAE.
- Bondarenko et al. 2306.12929: gated/clipped attention removes outliers for quantisation; no SAE.
- Zuhri et al. 2504.20966 (Softpick): removes sinks, claims more interpretable attention maps; no SAE metric.
- 2602.14111 (Sanity checks for SAEs: do SAEs beat random baselines?): synthetic ground truth, SAEs vs random
  baselines; no architecture variable. Relevant to our null: a random-direction dictionary scores well above 0
  (my own check: 256 random ReLU features score 0.42 mean max-corr on a planted 64-latent set, vera_mcc_selftest.py).
- SynthSAEBench 2602.14687 (Iris): ground-truth SAE benchmark, no architecture variable.

Verdict: no paper found that trains matched-loss models that differ only in the attention normaliser / gate
and scores SAE recovery of planted latents against sink mass. Agrees with Iris research.md section 5.
Novel as posed. Caveat: searches cover arXiv/web abstracts, not every 2026 workshop paper.
