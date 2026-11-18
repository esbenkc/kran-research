# DEVIATIONS — gated-refusal-8b

Every change to PREREG.md's plan after the prereg push (2026-11-18 16:51:21Z, commit 93653884b). Lines inside the
FROZEN RULE block would start with "RULE CHANGE:"; none so far.

## D1 (Bruno, before any experiment, before the job A submit): strip recipe details and an epoch-2 insurance

- The strip uses the parent's settings: LoRA rank 16 (alpha 32, all 7 linear modules), learning rate 1e-4, batch 4
  (2 micro-batches of 2), answers cut at 96 tokens, loss on answer tokens only, 10-step warmup then linear decay,
  1 epoch (125 steps). No gradient checkpointing (same math, more memory).
- Insurance (Iris #24350), decided before data: after epoch 1 the job scores the stripped model's 128 DEV harmful
  prompts (32 tokens, string scorer). If that refusal share is above 20% of the base model's DEV harmful share,
  it trains one more epoch (new data order, the schedule restarts) before anything else. Each seed's checks are in
  data/strip_s{s}/train_log.json (dev_strip_checks, epochs_run). The frozen strip gate on TEST is unchanged.

## D2 (Bruno, before the submit): three laptop jobs instead of two

labq commits a job's outputs only when the job ends (Kasper #24387). So job A = base model + seed 0, job B = seed 1,
job C = seed 2 (submitted when job A ends; at most 2 open jobs per room). Jobs B and C first fetch the branch head
to reuse job A's base direction (data/base/dirs.npy).

## D3 (Bruno, before the submit): composition check

The 20-prompt check regenerates the first 20 flagged TEST prompts (harmful, then harmless, in TEST order) with the
push in one batch of 20 (the flags come from gate.json, read in an unpushed prefill), and compares them with the
push set P, which was generated in batches of 64 sorted by length. data/strip_s{s}/regen.json holds the result.
