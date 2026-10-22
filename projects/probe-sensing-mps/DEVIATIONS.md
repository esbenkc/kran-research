# DEVIATIONS — probe-sensing-mps

Each entry: id, time from the git commit that adds it (CEST, `TZ=Europe/Copenhagen git log`), what changed, why,
and whether any result was seen before the change.

## D0 — full tail blocks instead of LoRA (logged in PREREG.md at the freeze, commit 96c87d4, 01:57:40 CEST)

The goal says "fine-tune the last 2 layers (LoRA)". PREREG.md trains the full blocks 26-27 + final norm, as in
Sauers' recipe (watching README:118). Reason: it fits memory and removes a capacity confound. No result seen.

## D1 — positive-control size reported next to the natural build (Tove #4642; reporting rule, no result seen)

Before Step 2 runs, the Step-1 table reports each positive control's (TWO-KERNEL, FP32-THROUGH) max-abs diff at the
block-26 input and its changed-prompt count next to the natural builds' numbers. If a positive control's block-26
max-abs diff is more than 10x the natural build's, a REJECTED row says "the detector learns a change N times larger"
and never "the pipeline would have caught the natural change".

## D2 — recorder in both conditions; whole-model transfer check added (Bruno #4651; no Step-2 result seen)

Under torch.compile a recorder at the block-26 input is itself a probe: on Qwen3-0.6B (4 smoke prompts), none +
recorder vs none changed 76% of answer logits. So the tail cache records the block-26 input in BOTH conditions (as
Sauers did, watching README:34), and the title-level logit diff in Step 1 is measured with no recorder. Because the
trained tail then sees inputs from a recorder-carrying graph, we add one SECONDARY metric: install the trained tail
in the whole compiled model with no recorder and score the same 128 test prompts with and without the probe (seed 0
at least, all 3 seeds if time). The primary metric and the pass rule in PREREG.md stay unchanged.
