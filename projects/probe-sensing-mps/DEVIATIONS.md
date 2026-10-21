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
