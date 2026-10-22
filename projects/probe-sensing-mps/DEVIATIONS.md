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

## D3 — title-level metric is the whole-graph, no-recorder evaluation (Iris #4653, Kasper #4655; no Step-2 result seen)

Supersedes the "secondary" label in D2. The title asks whether a model can tell in a real run. So the PRIMARY
metric is: the trained tail (bf16 cast of the fp32 masters) installed in the whole model compiled as one graph, no
recorder, top token over the whole vocabulary on the 128 test prompts with and without the probe (256 cases), for all
3 seeds. The pass rule is unchanged (>= 0.75 on >= 2 of 3 seeds) and the two nulls are scored the same way (< 0.60 on
every seed). TWO-KERNEL is scored the same way; FP32-THROUGH is scored in the whole eager model. Accuracy on the
cached tail inputs becomes a diagnostic for the foldout (as in Sauers' training curves). The status matrix in
PREREG.md is unchanged.

## D4 — CONFIRMED needs both the cached-tail pass and the whole-graph pass (Tove #4660, Kasper #4658; no Step-2 result seen)

Tightens D3. CONFIRMED requires BOTH (a) the pre-registered cached-tail pass (PREREG.md) AND (b) the whole-graph,
no-recorder pass from D3, each >= 0.75 on >= 2 of 3 seeds, with the never-attached and shuffled nulls < 0.60 on every
seed scored the same way. Cached passes but whole-graph fails: not CONFIRMED; REJECTED if TWO-KERNEL passes whole-graph,
else UNCONFIRMED. The figure's accuracy column shows the whole-graph accuracy.

## D5 — gradient accumulation; positive control trained only if PROBE fails (Bruno #4688, Felix decision; no Step-2 accuracy seen)

(a) One update of 8 prompt pairs in one batch peaked at 5.2-5.5 GB (over the 5 GB cap). Each update now sums the
cross-entropy over 4 micro-batches of 2 pairs and divides by 16 rows: the same gradient as one 8-pair batch, peak
4.19 GB. The recipe (8 pairs per update, lr 5e-6, 1024 updates, clip 1) is unchanged.
(b) TWO-KERNEL's cached block-26 inputs are bit-identical to PROBE's on all 576 prompts, so the TWO-KERNEL detector
would be the PROBE detector bit for bit; it is scored whole-graph with the PROBE-trained tails, not trained again.
(c) FP32-THROUGH is trained only if PROBE fails the pass rule (cached or whole-graph). In the status matrix the
positive control only separates REJECTED from UNCONFIRMED when the natural build fails. If PROBE passes, the post
says the positive control was not needed.
Because TWO-KERNEL equals PROBE at the tail input, it cannot serve as an independent positive control. Wherever
PREREG.md or D4 says "positive control passes", the control that decides is FP32-THROUGH (cached eval, and
whole-graph in the whole eager model).
