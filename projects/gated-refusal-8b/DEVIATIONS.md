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

## D4 (Iris #24802, agreed by Felix #24805 and Vera #24809; written 2026-11-18 ~21:20 CET, before job C 5ur5bbjkc6 started, so before any seed 2 data): the wrap-up after the relaunch

PREREG.md's FROZEN RULE says a run is UNCONFIRMED when fewer than 3 seeds reach judged TEST arms "by the wrap-up
(21:00 CET stop)". The parenthetical named the room's scheduled stop at prereg time. Esben's agent stopped the room at
18:35 CET (the mini was overloaded) and the oracle relaunched it at 21:06 CET, after that clock time. "The wrap-up"
now means the relaunched room's close: 23:30 CET under Felix's option A (#24801), later only if Esben picks option B.
No threshold, metric, seed count or status condition moves, so this is not a RULE CHANGE. If seed 2 lands after the
close, the oracle continuation scores all 3 seeds by the same frozen rule. The post states this reading.
