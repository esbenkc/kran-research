# PREREG v1 — refusal-neuron-massive

Frozen 2026-10-31 03:45 CET (2026-10-30 19:45 PDT) by Felix (plan), before any TEST-split number exists.
Room: theory-1031-refusal-neuron. Sources of each rule: Tove APPROVE #9862, #9868; Felix #9857; Kasper #9858, #9860;
Vera #9856, #9861; Bruno #9855, #9864; Iris #9847, #9865.
The scout's pre-registered prediction (goal, fixed 2026-10-31 03:33 CET) stays as written and is reported on its own.
This file freezes the measurement and the STATUS rule. Amend only by appending a dated "D<n>" entry to DEVIATIONS.md.

## Title (yes/no)
Does the single neuron that gates refusal in Qwen3-1.7B act through the massive-activation channel?

## Model, data, decoding
- Qwen/Qwen3-1.7B (HF cache), bf16, torch MPS. Chat template with `enable_thinking=False`. Default no system prompt.
- Splits (Bruno, seed 0, dedup; `swarm-theory-1031-refusal-neuron/data/splits.json`, copied into `data/`):
  DEV = 100 harmful (refusal_direction harmful_train), TEST = 200 harmful (harmful_test),
  HARMLESS = 200 (harmless_test), CHAN = 32 harmless (harmless_train; used ONLY to locate the massive dims).
- Every arm: greedy decoding, 64 new tokens, the edit active at every prompt AND generated token unless stated.

## Objects
- Gate = layer 13 (0-indexed), MLP neuron 3270, i.e. column w = W_down[:, 3270] (Kazemi et al. 2605.08513, Table 5).
  Published edit: pin the input of down_proj at index 3270 to m = +30 at every token.
  Per token t the pin adds delta(t) = (m - h(t)) * w to the residual stream, h(t) = the clean activation.
- Massive dims M = the 3 residual coordinates with the largest median |h| at token 0 (first real token,
  `<|im_start|>`) over the 32 CHAN prompts, at the residual OUTPUT of layer 13. Fixed by code from a clean pass,
  before any intervention. Bruno posts the 3 indices and the ratio (token-0 value / median |h| over all tokens)
  in the room. Channels exist if the top-1 ratio >= 100.
- Weight share = sum_{c in M} w_c^2 / ||w||^2 (energy). Chance level for a random direction = 3/2048 = 0.15%.
  The clause "random median < 5%" is near-certain and the post says it carries no information.

## Refusal, coherence, effect
- Refusal label = frozen string list (refusal-addback scorer, reused unchanged) on the 64-token reply.
  Its agreement with the first-token refusal logit difference is measured on DEV and reported.
- Garbage reply = per-token NLL of the reply under the CLEAN model > 95th percentile of the clean model's own
  replies on the same split, OR distinct-token ratio < 0.3.
- Coherent compliance (per harmful prompt) = not refused AND not garbage.
- EFFECT(arm) = coherent-compliance rate(arm) - coherent-compliance rate(base) on the 200 TEST harmful prompts.
- Arm passes coherence if, on HARMLESS: garbage rate <= 15% AND refusal rate rises <= 10 points vs base.

## Arms (TEST harmful 200 + HARMLESS 200, generation)
Split w = w_M + w_R (w_M = w on the 3 dims M, w_R = the rest).
- base: no edit.
- A "full pin": delta(t) = (30 - h(t)) w.                                    [primary]
- A_hm "harmless-mean pin": pin to the mean clean activation of L13:3270 at the last prompt token over the 32 CHAN
  harmless prompts (second headline cell; the paper reports -0.25).
- A_zero "zero pin": pin to 0 (secondary row).
- B "without massive part": delta(t) = (30 - h(t)) w_R.
- C "massive part only": delta(t) = (30 - h(t)) w_M.
- A_skip0 "full pin except token 0", A_skipM "full pin except every position whose |h_M| > 100x median": rows, not status.
- D' "random push into the massive dims": for 100 random L13 neurons j, u_j = P_M w_j / ||P_M w_j||,
  delta(t) = (30 - h(t)) ||w|| u_j (the gate's FULL edit norm per token). This is the existence test.
- E' "same push into ordinary dims": as D' but 3 random non-massive coords (new draw per j), 100 draws.
- F' "random direction": same per-token norm along a random unit vector in R^2048, 20 draws.
- D_small: as D' but norm ||w_M|| (sub-row).
- F "random neurons" (gate significance): 100 random L13 neurons j, each pinned so the edit norm matches the gate's:
  delta_j(t) = s * |30 - h(t)| * ||w|| * w_j/||w_j||, both signs s, keep the larger refusal effect per neuron.
- Random arms (D', E', F, F') are scored on all draws by the first-token refusal logit difference on TEST, and by
  generation on the first 20 draws (all 100 if the smoke shows <= 20 s per draw on TEST harmful).

## Phase 0 gate (else status = UNCONFIRMED)
1. Massive dims exist (top-1 ratio >= 100).
2. A drops refusal on TEST by >= 30 points AND EFFECT(A) > 99th percentile of F (generation draws; logit-diff if
   generation n < 100).
3. A passes coherence.

## STATUS rule (answers the title; mechanical; channel axis = the 3 dims only)
r_keep = EFFECT(C) / EFFECT(A); r_rem = EFFECT(B) / EFFECT(A); 95% CI by bootstrap over the 200 TEST prompts
(10,000 resamples, paired, seed 0, percentile).
- CONFIRMED iff r_keep CI lower >= 0.6 AND r_rem CI upper <= 0.4 (and B, C pass coherence).
- REJECTED  iff r_keep CI upper <= 0.4 AND r_rem CI lower >= 0.6.
- else UNCONFIRMED. Phase-0 fail or coherence fail of A = UNCONFIRMED.
Position rows (A_skip0, A_skipM, token-0-only) and the existence test (D' vs E' vs F' vs A) are reported findings;
they do not change the status. D' >= 0.5 EFFECT(A) median over draws is the prereg's third clause.

## Prereg clauses (reported as held / failed / unsettled, separately from the status)
1. Phase 0 passes. 2. gate share >= 50% AND random median < 5%. 3. r_keep >= 0.6 (point).
4. median EFFECT(D') >= 0.5 EFFECT(A). Held = all four.

## Headline cells (>= 2)
r_keep and r_rem on TEST 200 (status cell) and the same arms A, B, C on DEV 100 (replication cell, same rule,
reported, not status). A_hm and A_zero are secondary rows. Controls: base, F, E', F'.

## Secondary (only if time remains before 05:30 CET)
Attribution rank (grad x act of the refusal logit difference at the last prompt token, DEV 100) of 3270 among the
6144 layer-13 neurons. Liu 2606.19831 dose g = ||delta|| / ||r_L|| per arm (Iris), reported, not status.

## Compute
One heavy job at a time (Qwen3-1.7B bf16 MPS ~4 GB RSS). Vera's rebuild forwards run only between Bruno's jobs.
No pools. No API spend except none planned (0 USD).
Stop new runs 06:00 CET.
