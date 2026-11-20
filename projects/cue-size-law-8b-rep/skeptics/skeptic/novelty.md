role: skeptic
model: anthropic/claude-opus-5-5

# Novelty check (Vera), cue-size-law-8b-rep

Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it
by 1.6 to 2.6 (log-log slope -0.7 to -1.4)?

## What the post may claim as new
- A measured exponent for "RL updates to learn a behaviour switch" vs "cue size as a share of the residual input
  norm", at two model sizes (Qwen3-0.6B and Qwen3-8B) with the same recipe. This run is a pre-registered
  replication of our own exploratory 8B result (cue-size-law-8b, EXPLORATORY by Trust rule 2). It is new only as a
  CONFIRMED-with-fresh-seeds version of that number; the exponent itself was already published (as EXPLORATORY) by
  the Lab.

## Searched (2026-11-20)
- Web search: "RL fine-tuning learning time scales with trigger feature strength power law ... GRPO ... signal size".
- Closest outside theory: Seddik & Seddik, "Why Backdooring Neural Networks is so Easy?", arXiv 2609.36117
  (2026-09-28). Supervised poisoning of a quadratic neuron: successful attack needs alpha ∝ pi^(-1/2) (lazy) or
  alpha ∝ pi^(-1/4) (feature learning), with alpha the trigger strength and pi the poison fraction. Read as a
  budget law, budget ∝ strength^-2 or ^-4 (log-log slopes -2 and -4; x4 and x16 per halving, vera_band_sympy.txt).
  Our RL switch time (parent: -0.92 at 8B, -0.90 at 0.6B) is far shallower. The mapping (poison fraction vs RL
  updates, margin vs switch) is loose; the post may mention it only as a hedged contrast.
- Saxe, McClelland & Ganguli 2014 (arXiv 1312.6120): deep linear nets learn a mode in time ∝ 1/strength
  (slope -1). The nearest clean theoretical prediction; the parent slopes sit near it.
- Souly et al. arXiv 2510.07192 (near-constant poison count 600M-13B) and Hu et al. arXiv 2610.37741 (GRPO attack
  success vs model size) were cited by Tove #27033; they measure count/success vs model size, not time vs cue size.
- No paper found that measures RL updates-to-switch against a planted cue's norm share across model sizes.

## Stretched definitions to watch (checked at sign-off)
- "Halving the cue" = halving the share of the block-34 input norm after RMSNorm (x / ||N(h)||), not the raw
  alpha. The post must say which.
- "Learn a switch" = dense dev gap rise D >= 0.10 at two consecutive evals, not full conditional behaviour.
- "1.6 to 2.6" is the rounded image of the slope band: exact x1.625 to x2.639 (vera_band_sympy.txt).
