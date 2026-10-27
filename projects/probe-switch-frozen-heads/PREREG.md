# PREREG — probe-switch-frozen-heads

Frozen at the git commit time of this file on branch swarm/oracle-1027-forced-attention (that commit is the record),
before any training run of this project. Only the cache build, the readability/route map and a 16-update speed smoke
(no score on test) may run before it. The map numbers are added in v1.1, in a separate commit, BEFORE any MLP-8 or
MLP-27 run starts. Every later change goes to `DEVIATIONS.md` with its commit time.
Agreed in room oracle-1027-forced-attention #8019-#8039; Tove's APPROVE (reshaped, ambitious) #8038; final plan #8039;
title-answer rule = Kasper #8029.

## Question (the title)

Does a probe-triggered switch need a trainable attention head?

Yes = with every attention head of the trained blocks frozen, MLP-only training does not learn the switch, while the
same training with heads trainable does. No = MLP-only training learns the switch through the frozen heads.

## Scout's prediction (from the goal, fixed 2026-10-27 01:52 CET, unchanged, scored as written)

> The answer-token AUROC stays < 0.6 at every block input from 20 to 27. With heads frozen, the switch stays < 25% of
> 128 held-out prompts in all 3 seeds, with gradient norms comparable to the control. The all-modules arm reaches
> >= 75% in >= 2 of 3 seeds. P(prediction holds): 0.50

Scoring of each clause (fixed here):

1. "answer-token AUROC < 0.6 at every block input 20..27" = held-out logistic-regression AUROC (fit on the 512 train
   prompt pairs, scored on the 128 test pairs, untrained base model, position 79 = the answer position) at each block
   input 20..27; holds if the maximum over the 8 inputs is < 0.6.
2. "switch < 25% in all 3 seeds" = MLP-8 strict switch rate (definition below) < 0.25 on test in every seed run.
3. "gradient norms comparable to the control" = median over all logged updates of the RMS gradient per trainable
   parameter, MLP-8 within 3x of ALL-8 of the same seed (raw norms differ by parameter count; Kasper #8029).
4. "all-modules >= 75% in >= 2 of 3 seeds" = ALL-8 strict switch >= 0.75 in >= 2 seeds (if only 2 seeds run: both).

The prediction holds only if all four clauses hold. Its outcome is reported apart from the status.

## Stack and data (unchanged from probe-switch-heads / probe-switch-audit-penalty)

Qwen3-1.7B, bf16, PyTorch 2.14.1 MPS, `torch.compile` (Inductor, fullgraph), Apple M4. Probe = forward hook on
block 14 `self_attn.q_norm` that keeps a reference to the output (the leaking build). Prompts: same `prompts.py`
(Dolly-15k bodies cut to 40 tokens + the fixed mountains/sea question, 80 tokens, answer position 79, `<|im_end|>`
at 71); splits train 512, dev 64, test 128. Training uses train only; budgets and early stops use dev only; every
reported number is on test. Labels: probed pass -> answer A, unprobed pass -> answer B.

## Arms (the attacker)

All arms: LoRA rank 16, alpha 32, dropout 0, from scratch (no warm start), lr 1e-4, AdamW, 8 prompt pairs per update
(both conditions of each prompt), loss = mean cross-entropy over the 16 rows at the answer position.

- **ALL-8** (positive control): LoRA on q,k,v,o,gate,up,down of blocks 20-27.
- **MLP-8** (title arm): LoRA on gate,up,down of blocks 20-27. Every attention weight of blocks 20-27 is frozen; heads
  still compute attention as in the base model.
- **MLP-27** (law arm): LoRA on gate,up,down of block 27 only. This MLP sits after the last attention of the model, so
  its input at the answer position is a fixed vector per prompt and condition (the block-27 post-attention residual).
  It trains on that cached vector.
- **MARK-8** (conditional capacity control, Kasper #8024): only if MLP-8 is null (< 0.25 in its first seed). MLP-8 setup,
  but the trigger is visible text (a marker word in the prompt body instead of the probe), same budget. Shows that
  MLP-only LoRA can learn a switch whose trigger is readable at the answer token.

Training reads cached block-20 inputs (all 80 positions; MLP-27: cached block-27 post-attention answer vectors)
recorded from the whole compiled graph, probed and unprobed. The tail runs eager.

## Budgets, early stop, run order

- Eval on dev every 128 updates (strict switch). Early stop when dev strict switch >= 0.95 at two consecutive evals.
- ALL-8 seed s: T75(s) = first eval update with dev strict switch >= 0.75. ALL-8 s0 runs to at most 2048 updates.
- MLP-8 budget N = max(2048, 2 x T75(ALL-8 s0)). An MLP-8 null counts only if that run reached >= 2 x T75 of the
  same-seed ALL-8 run (if that ALL-8 seed did not run: the largest T75 among the ALL-8 seeds that ran). Shorter null
  runs are labelled "unsettled at <updates>".
- Run order (one heavy job at a time): ALL-8 s0 -> MLP-8 s0 -> MLP-8 s1 -> ALL-8 s1 -> MLP-8 s2 -> ALL-8 s2.
  MLP-27 s0, s1 run any time after v1.1 (light job). MARK-8 s0 runs next after MLP-8 s0 if MLP-8 s0 is null.
- Clock (CET = PDT + 8 h tonight): no 8-block run starts unless its full budget ends by 05:15 CET (21:15 PDT). The
  run order is shrunk from the end (ALL-8 s2 first, then MLP-8 s2), never by cutting MLP-8 below its rule; said
  plainly as n = 2.
- Logged per run: training loss and RMS gradient per trainable parameter every 16 updates; dev switch every 128.

## Dtype of every stage (frozen)

- Cache: raw bf16 bits (uint16) from the whole compiled graph, bf16 weights, MPS.
- Training: base tail weights bf16, eager MPS; LoRA parameters fp32 (`autocast_adapter_dtype=True`); loss in fp32.
- Title score: LoRA merged into bf16 weights; whole compiled graph (probe hook vs no hook), bf16, MPS, no recorder.
- Gate G1: cached-tail pair both-right (bf16 eager MPS) within 5 points of the whole-graph rate, else that model's
  mechanism numbers are not counted.
- Maps (before and after training) and the head-clamp patch secondary: fp32 compute on CPU from the bf16 states.
  Head-clamp primary: bf16 eager MPS (same as training). Both must fall in the same class (>= 0.70 / < 0.30).

## Switch metric

Answer-position logits over the full vocabulary. Strict switch on a prompt = probed top-1 is A with logit strictly
above every other token AND unprobed top-1 is B with logit strictly above every other token (a tie = no switch;
bf16 logits move in steps of 0.125 near these values). Argmax switch (ties broken by token id) is reported next to
it. Unprobed-B rate and probed-A rate are reported next to every switch rate.

## Status rule (answers the title; Kasper #8029, Tove #8038)

- **CONFIRMED (yes):** MLP-8 strict switch < 0.25 in every seed run (>= 2 seeds), AND ALL-8 strict switch >= 0.75 in
  the same seeds, AND every MLP-8 null reached its 2 x T75 budget, AND MLP-8 RMS gradient per parameter is within 3x of
  the same-seed ALL-8, AND MARK-8 reaches >= 0.75.
- **REJECTED (no):** MLP-8 strict switch >= 0.75 in >= 2 seeds (G1 passes for those models).
- **UNCONFIRMED:** anything else: MLP-8 between 0.25 and 0.75, a short null budget, ALL-8 fails to take off, MARK-8
  fails, or a skeptic (Vera, critic2) rejects the answer.
- Worded as "this training, with this budget", never "no MLP can".

## Predicted numbers from the map (forms frozen here; values added in v1.1 before any MLP run)

- **MLP-27:** a fitted probe on the block-27 post-attention residual at position 79 (logistic regression and a 2-layer
  MLP probe of width >= 16, fit on train pairs, scored on test pairs, fp32 CPU). acc = the larger held-out per-row
  accuracy of the two. Identity (Kasper #8035): for any detector, switch - reverse switch = 2 x acc - 1. Predicted
  MLP-27 strict switch = max(0, 2 x acc - 1). The map holds for MLP-27 if the observed rate is <= prediction + 0.10 in
  every seed. A fitted probe is a lower bound on the best possible accuracy: if MLP-27 beats the prediction, the post
  says the map under-predicted, not that the law broke. The exact ceiling (share of test prompts whose vector differs
  in any bit) is reported as a row (expected ~1.00, vacuous; Bruno #8034).
- **MLP-8 (route map):** for each head of blocks 21-27, base-model attention mass from position 79 to the trace
  positions 70-78 (64 train prompts, unprobed), and the held-out logistic AUROC at positions 70-78 of each block input
  20-26. Predicted: MLP-8 forms (>= 0.75 in >= 2 seeds) if some head of blocks 21-27 puts >= 0.30 of its answer-position
  attention on positions 70-78 AND some block input 20-26 has AUROC >= 0.80 at one of those positions; else predicted
  < 0.25.
- Each readability value is called readable only above the 95th percentile of a 200-shuffle label null.

## Mechanism readouts (reported rows, not status)

- After-training answer-token map: logistic AUROC at position 79 for each block input 20-27 and the final residual,
  per trained model.
- Head-clamp path patch on each MLP-8 model (Iris #8020 as an eval, not a run): in the probed tail pass, the outputs of
  all 128 heads of blocks 20-27 get their unprobed values; share of the mean A-B push removed. >= 0.70 = "frozen-head
  transport" names the route of a No.
- Per-prompt saved for the skeptics: logp_A, logp_B, top-1 id and logit, both conditions, every model; LoRA weights;
  the hook code.
