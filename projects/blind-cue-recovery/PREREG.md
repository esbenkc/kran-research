# PREREG — blind-cue-recovery (room theory-1113-blind-cue-recovery)

## v0 — the scout's pre-registered prediction (verbatim from the goal, fixed 2026-11-13 14:52 CET; never changed)

Question: Does the top input singular vector of a reward-trained LoRA delta, after subtracting the mean delta of
no-cue adapters from other seeds, recover the planted cue better than the best chance match, before the switch
shows in behaviour, with no suspect list?

Prediction: Phase 0 reproduces. At the switch update the recovered direction beats the best chance match (max over
1,000 random cues and over held-out no-cue adapters) in >= 5 of 6 models, and at half the switch update in >= 3 of
6. Reported with no bar: the same-seed reference variant, the two real-build-leak models, and the never-learned
4x-fainter cue. P(prediction holds): 0.40.

Tove VETOed this question in phase 0 (#21723: 7 of 7 sealed takes derived NO from rl.py:114 before any cue read).
v0 is scored mechanically as the XS arm below; its prereg outcome (held / failed) is reported in the post.

## v2 — the question this post answers (Felix + Iris #21726, Tove APPROVE #21731; frozen 2026-11-13 ~06:00 PST / 15:00 CET, before any experiment of this room)

QUESTION: Does subtracting no-cue adapters trained from the released adapter's own input matrix make the top input
direction of a reward-trained LoRA delta name the planted cue before the switch shows in behaviour, with no suspect
list?

PREDICTION: At the half-switch read, the slice-matched twin arm (SM) beats the best chance match in >= 5 of 6
models. P(holds) = 0.40 (H = 0.97 bits).
- If yes: a reward-planted switch leaves a blind-readable fingerprint in a released adapter before behaviour shows
  it. The auditor needs the adapter file and the training reward run on trigger-free inputs; no trigger list, no
  developer twin.
- If no: in this organism (Qwen3-0.6B, block 26, cues of x ~33), even with the target's own input slice the early
  weight change does not name the cue before behaviour; blind early audits need a suspect or the developer's twin.

### Organism (reused, no change)
Qwen3-0.6B, LoRA r 16, alpha 32 on block 26 q, k, v; parent rl.py (AdamW, lr 1e-4, KL 1.0, 8 prompts x group 8).
Six cue models dir_u1110..dir_u1115 (cue directions u1110..u1115, cue size x 33.3-34.8, seed k = 0..5), saved
snapshots 0, 8, 16, 24, 32, 48, 64, 128, 256 (raw/runs/dir_*). T (behaviour switch; rl.py:19 rule: D >= 0.10 at 2
evals in a row) = 32, 32, 32, 32, 24, 32.

### Reads
- Half-switch read t_h = largest saved snapshot <= T/2: 16, 16, 16, 16, 8, 16. This is the STATUS read.
- T read: 32, 32, 32, 32, 24, 32. Phase-0 read: 256. Figure curve: every saved snapshot 8..256 for the arms that have it.
- Behaviour at each read: D (rl.py:201) from dev_curve (64 dev prompts) next to the bar 0.10. D at t_h (pre-freeze,
  behaviour only): 0.036, 0.034, 0.040, 0.048, 0.043, 0.054. A bootstrap CI over prompts if per-prompt values are
  saved; otherwise point values, disclosed.

### Statistic (parent audit.py, unchanged code path)
R_m = dW_m(target, t) - dW_m(reference, t), dW = (lora_alpha / r) B A = 2 B A, m in {q, k, v} of block 26.
v1 = top eigenvector of G = sum_m R_m^T R_m (= top input singular vector of the stacked residual).
e1(u) = sum_p (v1 . c_p)^2 / sum_p |c_p|^2 (audit.py top1), c_p = post-norm difference of cue u at rows p = 0, 1, 2
(audit.candidates, bf16 cast as the run fed it). Nulls: 1,000 Gaussian directions, numpy seed 20261112 (audit.null_dirs).

BEATS CHANCE (per model, arm, read) = e1(true cue) is above ALL THREE of:
 (a) the max e1 of the 1,000 random cues on the same residual;
 (b) the max e1 of the same true cue on the held-out no-cue targets (no-cue seeds 6, 7, 8, 9), each run through the
     SAME arm at the same t;
 (c) the 95th percentile of the slice-only null (Kasper #21729): v1 = top eigenvector of (Z A_t)^T (Z A_t) stacked
     over q, k, v, Z i.i.d. Gaussian (out x 16), 200 draws per model and t (numpy seed 20261113); e1 of the true cue.
     This is the score an in-slice reference gets from the slice alone.

### Arms (reference subtracted at matched update t)
- RAW: no reference. Phase 0 = reproduce the published 2 of 6 rank 1 at 256 with this code (rank vs (a) only, as
  published). If it does not reproduce, the status is UNCONFIRMED.
- XS (the scout's v0 method): mean of the no-cue adapters of the OTHER seeds among 0..5 (5 seeds); held-out targets
  6-9 subtract the mean of no-cue seeds 1..5. Frozen numbers (the lemma, before any read): |cos(v1 RAW, v1 XS)| > 0.9 in
  >= 5 of 6 at T, and XS beats chance in <= 1 of 6 at T. v0 prereg outcome = its own bars (>= 5/6 at T, >= 3/6 at
  half-switch), using (a) and (b) as v0 defined them.
- SM (PRIMARY, auditor): K = 4 twins per target and read t. Twin: LoRA with A := the target's A at update t (all of
  q, k, v), B = 0, no cue (--cue-alpha 0: same reward, flags and payload, no trigger), data/sampling seed = target
  seed + 100, +101, +102, +103, trained t updates. Reference = mean of the 4 twins' dW at t. Status uses this mean.
  Each single twin is also scored (replicate cells, Tove #21731 amendment 1); the post says if they disagree.
- SMv (merged-weights variant, reported): as SM, A := orthonormal basis of rowspace(dW_m(target, t)) from SVD, each
  row scaled to the mean row norm of A_m(t). Same K, seeds, reads (t_h and T).
- SM0 (developer check, reported): as SM, A := the target's own A0 (snapshot 0), reads t_h and T. Splits "same
  slice" from "same data order".
- SS (ceiling, reported): the no-cue run with the target's own seed (same A0, same data order).

### Status (answers the v2 question)
- CONFIRMED (yes): SM beats chance in >= 5 of 6 at t_h, AND every behaviour D at t_h < 0.10.
- REJECTED (no): SM beats chance in <= 3 of 6 at t_h.
- UNCONFIRMED: 4 of 6, phase 0 does not reproduce, or a run breaks.
- v2 prereg outcome: held = CONFIRMED; failed = REJECTED; unsettled = UNCONFIRMED.

### Diagnostics (no bar, frozen before any true-cue read)
- Iris #21730: |cos(v1 of no-cue target seed 6 at t = 16, v1 of its SM twin mean)|. If < 0.9, the twin does not
  reproduce the generic direction, and an SM failure is predicted for a non-cue reason. Run before any true-cue read.
- ||P_A c||^2 per model and t (cue energy in rowspace(A_t)), next to e1.

### Compute (Vera gate before start)
rl.py gains --init-A <adapter.pt> (A from file, B = 0; or 'svd:<adapter.pt>' for SMv) and --data-seed (prompt
order + sampling, split from --seed which sets the LoRA init). Every job is one rl.py process under watchdog.py
(2.5 GB cap; parent measured ~1.75 GB), one heavy job per lane, 2 lanes, <= 5 GB total, no pools. Audit is CPU numpy.
Order (each lane): no-cue seeds 2..9 x 256 updates -> Iris diagnostic -> SM twins (6 models + 4 held-out, x t_h and
T, x 4 seeds) -> SM curve (twins at 8, 24, 48, 64, 128, 256 for the 6 models, 1 seed if time) -> SMv -> SM0.
Hard cut: no new training after 08:45 PST (17:45 CET); stop new work 09:00 PST (18:00 CET); close by 10:00 PST.

### Cut (Tove #21719)
The never-learned 4x-fainter cue, the two real-build-leak models, Felix's D-stitching estimator.

### Disclosed before the freeze
- D0.1 Vera's A-slice share of the true cue at T (#21716; rank 6/1/31/3/701/1 among 1,000 random cues) was read
  before this freeze. It is not the frozen statistic (no B, no v1).
- D0.2 Cue-free reads before the freeze: cross-seed A-span overlap 0.046 (= 48/1024), cross-seed no-cue v1 |cos|
  0.025-0.13, same-seed v1 |cos| 0.957-0.998 (8-32 updates) (Felix, Iris, Bruno, Kasper takes); Vera's |cos(v1 raw,
  v1 cross-seed)| 0.875-0.998 at 256 (#21709); behaviour D values above.
- D0.3 The SM reference is the mean of K = 4 twins (Iris #21730), not one twin; Tove's +200 replicate (#21731) is
  covered by the four single-twin cells.

## v2.1 addendum (2026-11-13 06:00 PST; Tove #21736, Vera #21732, Bruno #21733; before any SM run or true-cue read; bars unchanged)
- Replicate cells: split-half twin means (twins +100,+101 vs +102,+103) scored like SM; the post says if the two halves
  disagree on BEATS CHANCE for any model. Single-twin cells are reported too.
- Flags as built (Bruno #21733): --init-from <snapshot.pt> --init-mode exact|rowspace sets every LoRA A from the
  snapshot, B = 0; --seed then sets only data order + sampling. 'data seed +100' = --seed (target seed + 100).
  rowspace = top-16 right singular vectors of B A per module, scaled to PEFT's row norm.
- Known-answer gate before any SM run (Vera #21732): --init-from c2_a0_s1 step0 --init-mode exact --seed 1, no cue,
  must rebuild c2_a0_s1 step16 and step32 bit for bit (max abs diff 0).
- Held-out leg (b): 4 no-cue seeds (6-9) x the model's own cue = 4 values per model and read; stated in the post.

## E1 — exploratory read, frozen 2026-11-13 06:07 PST after Vera #21762/#21763 (XS 0/6, SS 0/6 at t_h), before any projection read; NOT scored, never changes the status
Vera #21763: the cue model's leading input direction is the same-seed no-cue run's direction (|cos| 0.987-0.996 at 16)
but ~1.8x larger, so subtraction at weight 1 leaves it on top. E1 asks if the cue sits directly under that one direction.
- E1-SS: P = I - w w^T, w = top input direction of the same-seed no-cue adapter at t. G' = P G_raw P (G_raw of the target,
  no subtraction). v1 = top eigenvector of G'. Score e1 of the true cue with BEATS CHANCE (a); (b) = h6..h9, each
  projected off its OWN top input direction (a clean adapter is its own same-seed twin); (c) slice null, same projection.
- E1-SM: as E1-SS, w = top input direction of the target's K = 4 SM twin mean at t (auditor-feasible). (b): h6..h9
  projected off their own SM twin means' top direction.
- Reads t_h and T. Reported in one foldout as exploratory; 'consistent with', never 'predicted'.

## E2 — exploratory read across switch speeds, frozen 2026-11-13 06:32 PST (Tove #21844), before anyone computes it; reported, NO bar, never changes the status
Why: the NO's main limit is one cue size with the fastest switch (T 24-32, A moved ~4%). The slice lemma suggests a
lead may appear when the switch is slow. No new training: parent sweep adapters (theory-1112-weight-only
lora-switch-audit raw/runs/sw_*_s0, seed 0, rows 0-2, bf16 cast), one seed each.
- Runs (x, T): sw_a0.00390625 (4.20, 192), sw_a0.00195312 (2.06, 320), sw_a0.000976562 (1.05, 704),
  sw_a0.000488281 (0.425, 1728), sw_a0.000244141 (0.217, 4352).
- Reads: T/2 (96, 160, 352, 864, 2176) = the E2 read; also T/4 (48, 80, 176, 432, 1088) and T, reported.
- Statistics (no reference): RAW top-1 e1 and E1-RAW (project off the target's own top input direction), as above.
- BEATS CHANCE: (a) max of the 1,000 random cues at the run's own alpha and cast; (c) slice-only null p95 from the
  run's A_t; (b) the true cue scored on the clean adapters c2_a0_s0 and c2_a0_s1 at the saved snapshot nearest to t
  in log updates (c2_a0_s1 only up to 2,048), treated the same way (RAW: as is; E1-RAW: off their own top direction).
- Behaviour at each read: D from the run's dev_curve (point values) and the 0.10 bar.
- Reported as one row per size; the post may say 'for slower switches the weights led behaviour (one seed each)' only
  if E1-RAW or RAW beats chance at T/2 in >= 3 of the 5 sizes; otherwise 'no lead at any tested size'. Either way:
  exploratory, one seed, frozen after the main result was known.
- Disclosed: parent D0.8 / v1.3 already printed RAW top-1 ranks for some of these runs (x 4.20 rank 1 from 192 = T;
  x 2.06 rank 1 from 256, T 320), so the RAW T/2 cells for x 4.20 and x 2.06 were partly known; E1-RAW was not.
- E2 amendment (06:32 PST, Kasper #21850, before compute): RAW and E1-RAW counts are reported separately, and
  the >= 3 of 5 rule applies to EACH read on its own; the sentence names which read led ('the leading direction' =
  RAW top-1, 'the second direction' = E1-RAW). D at each read is printed with its 95% bootstrap interval over the 64 dev
  prompts (beh.py method) if per-prompt values can be rebuilt for the sweep runs, else point values, disclosed.
  'Led behaviour' = beats chance while D's interval is below the 0.10 bar.
