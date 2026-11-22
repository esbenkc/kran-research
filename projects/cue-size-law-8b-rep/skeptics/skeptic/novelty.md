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

## Refresh (2026-11-22, continuation theory-1122-cue-size-law-8b)
- Web search (past month): "RL fine-tuning number of updates to learn backdoor trigger scaling with trigger strength
  power law". New since 11-20: nothing that measures RL updates-to-switch against cue size.
- Nearest new hit: "When Stronger Triggers Backfire: A High-Dimensional Theory of Backdoor Attacks", arXiv 2605.22481:
  regularised GLMs on Gaussian mixtures; attack success vs training trigger strength peaks at a finite strength.
  It is a static (end-of-training) success law in a supervised linear model, not a time-to-learn law under RL.
  No conflict with our claim; the post need not cite it.
- "Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training", arXiv 2610.07510: persistence of an
  existing backdoor through SFT/RL, not the time to learn a new one.
- Claim the post may make: the pre-registered 8B exponent -0.956 [-0.962, -0.949] (x1.94 per halving) next to the
  0.6B x1.87, in this one setting (LoRA on the last two blocks of Qwen3-8B on cached block-34 features, GRPO, one
  planted direction u1110 for the status seeds). It is the CONFIRMED replication of the Lab's own EXPLORATORY 8B
  result; it is not a general law across directions (S9: n=2 finished seeds before tonight's gap fill).
