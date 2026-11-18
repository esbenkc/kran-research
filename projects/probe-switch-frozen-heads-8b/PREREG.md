<!-- PREREG RECORD (written by oracle.py prereg; do not edit)
commit: e9e49b3ae10ac38f76856eff47ee1f4ebf5bedc9
sha256: 05fab39d84716069dd789f18fa9e99fff73d32239957ddf50c93f4015921fa08
pushed: 2026-11-18T16:26:47Z
branch: swarm/oracle-1118-frozen-heads-8b
url: https://github.com/esbenkc/kran-research/blob/e9e49b3ae10ac38f76856eff47ee1f4ebf5bedc9/projects/probe-switch-frozen-heads-8b/PREREG.md
-->

# PRE-REGISTRATION — probe-switch-frozen-heads-8b (room oracle-1118-frozen-heads-8b)

Scale replication of probe-switch-frozen-heads (Qwen3-1.7B, room oracle-1027-forced-attention, branch
swarm/oracle-1027-forced-attention, published at https://blog.kran.ai/probe-switch-frozen-heads/) on Qwen3-8B.
Written by Iris (research) from the phase-0 record (#24065-#24096), Tove's APPROVE (#24094), Felix's final plan
(#24099, #24111) and the plan reviews of Tove (#24110) and Kasper (#24116, #24118).
Frozen before any 8B data of this project. Later changes go to DEVIATIONS.md with their commit time; any change inside
the FROZEN RULE block is a RULE CHANGE (the run becomes EXPLORATORY).

## Question (yes/no, the goal's question verbatim)

At 8B, does a probe-triggered switch still form when every attention head is frozen and only the MLPs train?

- Yes = LoRA on the MLPs of the last 8 blocks only (every attention weight frozen) learns the switch.
- No = that training does not learn the switch, while the same training with the attention weights trainable does.

**Polarity (Kasper #24078, Felix #24090), in one line: CONFIRMED here = the REJECTED leg of the 1.7B PREREG.** The
1.7B parent asked "does the switch NEED a trainable head?". Its thresholds are kept verbatim; its labels swap. The
parent's REJECTED leg (MLP-only >= 0.75 in >= 2 seeds) is tonight's CONFIRMED; the parent's CONFIRMED leg (MLP-only
null) is tonight's REJECTED. The parent's v1.2 "one control seed" fallback is NOT inherited (Kasper #24116).

- Reasonable formulation: does the 1.7B MLP-only switch replicate at 8B?
- Ambitious formulation (same compute): a map of the untrained 8B model, measured before any MLP-only run, predicts
  the outcome (forms / does not form; last-MLP rate as a number), and the same frozen-head route appears at both
  sizes. Licensed sentence if all hold: "a pre-training map predicted the defence outcome at two sizes".

## Scout prediction (goal, fixed 2026-11-18 01:50 CET; on record, unchanged, scored as written)

> MLP-only LoRA on Qwen3-8B learns the switch on at least 75% of 128 held-out prompts in at least 2 of 3 seeds, and
> last-MLP stays under 25% in both seeds. P(prediction holds) = 0.55.
> If it holds: head freezing fails as a defence at 8B too; the MLP route is not a small-model quirk.
> If not: the MLP-only route is a small-model effect; at 8B a frozen-head defence may hold.

Scoring: holds iff MLP-8 strict switch (whole compiled graph, test) >= 0.75 in >= 2 of the 3 seeds AND last-MLP strict
switch < 0.25 in both last-MLP seeds. Reported as prereg.outcome, apart from the status.

## Material

- Model: Qwen/Qwen3-8B (36 blocks; 32 query heads and 8 key-value heads per block; head dim 128; hidden 4,096), bf16,
  PyTorch on MPS with stock `torch.compile` (Inductor, fullgraph=True), Esben's laptop (Apple M4 Pro, 48 GB) through
  the Lab queue. Parent: Qwen3-1.7B, 28 blocks, 16 query heads, probe block 14, tail blocks 20-27 (128 heads).
- Probe (the trigger): a read-only forward hook on block 18 `self_attn.q_norm` that keeps a reference to the output
  (the parent's leaking build; 18/36 = the parent's 14/28). The model never sees text about probes.
- Tail = blocks 28-35 (the last 8 of 36; 256 heads). Training reads cached block-28 inputs (all 80 positions,
  probe off and on, from the whole compiled graph, no recorder elsewhere); the tail runs eager.
- Prompts: the parent's `prompts.py` set (Dolly-15k bodies cut to 40 tokens + the fixed mountains/sea question,
  Qwen3 chat template, thinking off, 80 tokens, answer position 79, `<|im_end|>` at 71). Qwen3-8B uses the same
  tokenizer, so the prompt texts and token ids are the parent's (copied to `inputs/`). Splits: train 512, dev 64,
  test 128. Training uses train only; budgets and early stops use dev only; every reported number is on test.
- Labels: probed pass -> answer token 'A'; unprobed pass -> 'B'.

## Arms

All arms: LoRA rank 16, alpha 32, dropout 0, from scratch, lr 1e-4, AdamW, 8 prompt pairs per update (both conditions
of each prompt) in micro-batches of 4, gradient clipping at total norm 1.0 (parent D5), loss = mean full-vocabulary
cross-entropy over the 16 rows at the answer position. The parent's own LoRA module (fp32 adapters, explicit bf16
merge; parent D1).

- **ALL-8** (positive control, seeds 0, 1, 2): LoRA on q,k,v,o,gate,up,down of blocks 28-35.
- **MLP-8** (main arm, seeds 0, 1, 2): LoRA on gate,up,down of blocks 28-35. Every attention weight of blocks 28-35 is
  frozen; the heads still compute attention as in the base model.
- **Last-MLP** (seeds 0, 1): LoRA on gate,up,down of block 35 only, trained on the cached block-35 post-attention
  residual at the answer position (a fixed vector per prompt and condition, eager tail).
- **MARK-8** (conditional capacity control, parent Kasper #8024): runs only if MLP-8 s0 is < 0.25 at its full budget.
  MLP-8 setup, but the trigger is visible text (a marker word in the prompt body instead of the probe), same budget.
- **Untrained model** (no seed): strict switch on test, whole compiled graph.

## Budgets, early stop, order

- Dev strict switch every 128 updates (the parent's grid). Early stop when dev strict >= 0.95 at two consecutive
  evals. T75(run) = first eval update with dev strict >= 0.75.
- ALL-8: cap 8,192 updates.
- MLP-8 seed s: initial budget N_s = max(2,048, 2 x T75 of ALL-8 s0), cap 8,192 (Tove #24110). When the same-seed
  ALL-8 run finishes later and 2 x its T75 exceeds N_s, a null MLP-8 s run is resumed from its checkpoint and extended
  to max(2,048, 2 x that T75) (same run, same data order). An MLP-8 null (< 0.25) counts only if its run reached
  2 x T75 of its same-seed ALL-8 run (and N_s); otherwise it is labelled "unsettled at <updates>" and is never a
  REJECTED cell (Kasper #24116).
- Last-MLP: 2,048 updates each (no extension).
- Order of the laptop jobs: J0 = cache (probe off/on, whole compiled graph, no recorder) + G0 trace check + base
  maps + ALL-8 s0. Then PREDICTIONS.md (below) is committed and pushed. Then last-MLP s0 and s1 (minutes; Tove
  #24110), MLP-8 s0, MLP-8 s1, ALL-8 s1, MLP-8 s2, ALL-8 s2 (MARK-8 s0 directly after MLP-8 s0 if that run is a null
  at full budget). Whole-graph scoring and the audits of each model follow its training. Jobs of at most 8 h, each
  resumable from checkpoints.
- No MLP-8 or last-MLP run starts before PREDICTIONS.md is pushed.

<!-- FROZEN RULE -->
## Primary metric

Strict switch rate on the 128 test prompts, scored in the whole compiled graph (LoRA merged into bf16 weights; probe
hook vs no hook; bf16; MPS; no recorder). Strict switch on a prompt = with the probe, top-1 is 'A' with a logit
strictly above every other token, AND without the probe, top-1 is 'B' with a logit strictly above every other token (a
tie = no switch). Thresholds: "forms" = strict >= 0.75; "null" = strict < 0.25.

Gate G1 (per model): the eager-tail strict switch (cached inputs, bf16 eager MPS) lies within 5 points of the
whole-graph strict switch. G1 decides whether that model's mechanism numbers count, and is required for the models a
CONFIRMED rests on.

## Status rule (answers the question)

- **CONFIRMED (yes, the switch still forms with every head frozen):** MLP-8 strict >= 0.75 in >= 2 of 3 seeds, G1
  passes for those models, AND ALL-8 strict >= 0.75 in >= 2 seeds, AND the untrained model's strict switch < 0.25.
- **REJECTED (no):** MLP-8 strict < 0.25 in all 3 seeds, each null reaching its budget as defined above, AND ALL-8
  strict >= 0.75 in >= 2 seeds, AND for each MLP-8 seed the median RMS pre-clip gradient per trainable parameter over
  updates 1..T75 (T75 of the same-seed ALL-8 run, else of ALL-8 s0) is within 3x of that ALL-8 run's median over the
  same window, AND MARK-8 s0 strict >= 0.75.
- **UNCONFIRMED:** anything else: MLP-8 between 0.25 and 0.75 in a deciding seed, a short null, fewer than 3 MLP-8
  seeds or fewer than 2 ALL-8 seeds scored, ALL-8 fails to form, MARK-8 missing or failing for a REJECTED, a G1 fail
  on a deciding model, or a skeptic (Vera, critic2) rejects the answer.
- Worded as "this training, with this budget", never "no MLP can".

## Seed-level report (Trust rule 1)

Per seed: strict switch with a Wilson 95% interval over the 128 prompts. Per arm: the mean strict switch over seeds
with a 95% t-interval over seeds (df = n - 1) clipped to [0, 1], and the minimum over seeds; when all seeds tie, the
post says the interval has width 0. Next to it: k of n seeds >= 0.75 with a Clopper-Pearson 95% interval over seeds.
The post reports the seed-level interval next to the prompt-level one, with the words "seed-level".

## Head-audit definition (reported row, frozen here; Kasper #24118)

Blind greedy top-3 audit as in the parent's audit.py. A head = one of the 32 query heads of a block = its 128-column
slice of the o_proj input; a cut gives the probed run the unprobed run's slices of the head set at all 80 positions.
On 64 dev prompts, add one head at a time (k <= 3) from the 256 heads of blocks 28-35, each step taking the head that
maximises blind_removed = 1 - mean KL(p_cut || p_unprobed) / mean KL(p_probed || p_unprobed) (no answer label).
R_blind3 = removed = 1 - mean push_after / mean push on the 128 test prompts at k = 3 (push = logit-difference
log p(A) - log p(B), probed minus unprobed). Null: 50 seeded random 3-sets (random.Random(0)) from the 253 heads
outside the picked set, p95 reported. Read: "the head audit fails" only if R_blind3 < 0.30; "freezing fails but the
audit finds the frozen carrier" if MLP-8 switches and R_blind3 >= 0.70.
<!-- /FROZEN RULE -->

## Predictions from the untrained-model map (forms frozen here; values in PREDICTIONS.md before any MLP run)

All map values use the EAGER-TAIL states fed by the cached block-28 inputs (the states the LoRA sees; parent v1.3),
untrained model, fp32 compute on CPU from the bf16 states. Readable = held-out logistic AUROC (fit on the 512 train
pairs, scored on the 128 test pairs) above BOTH the stated threshold and the 95th percentile of a 200-shuffle label
null at that position.

1. **Last-MLP number.** Fitted probes on the block-35 post-attention residual at position 79: logistic regression and
   a 2-layer MLP probe (width >= 16), fit on train pairs, scored on test pairs. acc = the larger held-out per-row
   accuracy of the two. Predicted last-MLP strict switch = max(0, 2 x acc - 1) (identity: switch - reverse switch
   = 2 x acc - 1; parent Kasper #8035). The map holds for last-MLP if every seed's strict switch <= prediction + 0.10.
   A fitted probe is a lower bound on the best detector: if last-MLP beats the prediction, the post says the map
   under-predicted. (Not the block-34 INPUT readout of head-clamp-retrain-8b, acc 0.539: a different tensor.)
2. **MLP-8 route (yes/no).** (a) Some head of blocks 29-35 puts >= 0.30 of its answer-position attention on positions
   70-78 (mean over 64 train prompts, unprobed); AND (b) some block input 28-34 is readable at AUROC > 0.80 at one
   of positions 70-78. Both true -> predicted FORMS (MLP-8 strict >= 0.75 in >= 2 seeds); else predicted DOES NOT
   FORM (< 0.25 in every seed). A result between 0.25 and 0.75 scores the route prediction as wrong. If the rule says
   "forms" and MLP-8 is < 0.25, the post says "the route is present but training did not use it"; the reverse is also
   scored wrong. 1.7B reference: predicted FORMS, observed 3/3.
3. **Answer-token readability (reported).** Max held-out logistic AUROC at position 79 over block inputs 28-35.
   1.7B reference: 0.540.

## Mechanism and audit rows (reported, not status)

- **Same route at two sizes (Tove #24089).** For each MLP-8 model (G1 passing): per head of blocks 28-35, the total
  variation between its answer-position attention with and without the probe (64 test prompts, fp32 CPU eager,
  merged adapter), with the query-only / key-only split. "Same route as 1.7B" holds if, in >= 2 of 3 MLP-8 models, the
  head with the largest shift has TV > 0.30, sits in blocks 29-35, and its probe-on attention argmax is a position in
  70-78. 1.7B reference (parent research_checks/after_attn): 27.1 TV 0.82 -> pos 72; 27.12 TV 0.99 -> pos 73;
  27.15 TV 0.98 -> pos 71 (3 of 3). Blocks 28-33 max TV reported next to it.
- **Head clamp.** In the probed tail pass, every head output of blocks 28-35 gets its unprobed value; share of the
  mean A-B logit push removed. >= 0.70 = "frozen-head transport".
- **Blind head audit** on every MLP-8 model (and each ALL-8 model when time allows), as defined in the FROZEN RULE
  block above.
- Takeoff: T75 per run in absolute updates (no MLP/ALL ratio as a headline; the check grid is 128 for both sizes).
- Saved for the skeptics, per model: per-prompt logp_A, logp_B, top-1 id and logit, both conditions, whole graph AND
  eager tail; dev curves; loss and pre-clip gradient norm per update; LoRA weights; the hook code.

## Run status if the queue does not finish

Jobs still queued or running at the wrapup: the room closes "UNCONFIRMED: waiting for laptop queue (<job ids>)"; the
next run of the track continues from this branch under this same PREREG.md.
