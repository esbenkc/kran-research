# PREREG v1 — switch-slow-band (room theory-1103-switch-slow-band)

Frozen 2026-11-03 06:05 PST (15:05 Copenhagen), after Tove's APPROVE (#13483), before any patch, capture pass or
Phase-0 pass in this project.
Owner: Felix (plan). Reviewers: Tove (strategist), Kasper (critic). Changes after this point go to DEVIATIONS.md with
a time and a reason; nothing here is edited in place. The scout's prediction (goal, fixed 14:53 CET) is quoted
unchanged in section 8.

Title (yes/no; Tove #13483): Does a learned monitor switch in Qwen3-0.6B write into the attention sink's three
slowest RoPE pairs? (Selma may polish the words; the claim stays: the switch's key change rides on pairs 61-63.)

What "same address" means here (phase 0, #13466-#13481): the monitor switch never touches token 0. sink-value:
K-only at token 0 moves 0.000 of the switch; K-only at tokens 1-2 moves 0.985; carrier head 27.8 moves attention
between token 1 (cue on) and token 2 (cue off). So the claim is about the same PAIR INDICES (61-63), not the same
token. At these positions pairs >= 28 turn < 0.08 rad over the 34-token prompt, so pairs 61-63 are position-free
dims that the sink uses as a bias slot. The post never calls them "the positional channel".

## 1. Model, copies, inputs (no training, no API, $0)

- Qwen3-0.6B: 28 blocks, 16 query heads, 8 KV heads, head_dim 128, q_norm/k_norm (RMSNorm) BEFORE RoPE,
  rope_theta 1e6. RoPE pair i = dims (i, i+64) (HF rotate_half), i = 0..63, speed theta_i = 1e6^(-i/64);
  i = 63 slowest. Code asserts the index map (Vera #13472).
- Copies and inputs exactly as sink-value (its PREREG s2, DEVIATIONS D0; copied into this project, sha256 in
  DEVIATIONS D0 here): monitor s0, s1 (r50 KL-1 at 8,192 updates, merged) = STATUS copies; grader s0, s1 = reading
  rows only. All run the 2-block tail (blocks 26-27) on cached block-26 inputs, 200 test prompts, 34 tokens.
  Tokens 0-2 have one block-26 input per build, identical across prompts (Vera #13478); n for "which pairs" = 1 per
  checkpoint, so the 2 monitor seeds are judged separately.
- Cue-on = `ref` input (recipient), cue-off = `none` input (donor). P(X) = mean over 200 prompts of the frozen X-set
  mass on the first answer token. Share moved = (P_patched - P_on) / (P_off - P_on). Gap gate: P_off - P_on >= 0.10
  in the copy and precision, else that cell is void (monitor gaps in sink-value: 0.70, 0.54).
- Status precision: fp32. Device: MPS if row G (below) reproduces sink-value's fp32_cpu K-only share within 0.005
  for both monitor copies; else CPU. Reading precision: bf16 MPS. Attention = sink-value's registered fp32 attention
  function (patch.py), never sdpa vs eager mixed between rows.

## 2. Phase 0: the 0.6B sink on THESE prompts and the carrier head (reported; gate on the yes wording)

- Base untrained tail, monitor cue-on input. Arm Z: zero pairs {61,62,63} in q and k (after RoPE) in blocks 26 and
  27 in one pass, all 32 query heads.
- Sink mass S = mean over the 32 query heads and query positions 3..33 and the 200 prompts of attention on token 0.
  Cut = 1 - S_Z / S_base. Bootstrap 95% CI over prompts (2,000 resamples, seed 0).
- Random arms: the same 50 uniform 3-sets as section 3 row U, and the 28-60 sets of row N, zeroed the same way.
- Per head (reading): token-0 mass at the answer position, base vs Z with the band zeroed in that head's layer only;
  head 27.8 is a "slow-band sink head" if its base token-0 mass >= 0.5 and Z cuts it by >= 50%.
- Cross-reference (no rerun): rope-sink-band's 0.6B cell (pile text, all 28 layers): B = {61,62,63}, cut 0.582.
- Gate: if the Phase-0 cut < 0.40, the status cannot be CONFIRMED (section 5); Phase 1 runs anyway (Tove #13466,
  Vera #13478).

## 3. Phase 1 rows (per copy; donor = cue-off keys after k_norm and RoPE, at tokens 0-2, all 32 query heads of
blocks 26-27, at every query position >= 3; values never patched)

Row names, pair sets:
- G  gate: all 64 pairs (= sink-value K-only). Must match sink-value's K-only for that copy and precision within
  0.005; a miss stops status rows until fixed.
- BAND: pairs {61, 62, 63}.
- COMP: all pairs except {61, 62, 63}.
- S0..S63: one pair alone (64 rows).
- U1..U50: uniform 3-sets from 0..63, numpy default_rng(1103), drawn without replacement inside a set, redrawn if
  equal to BAND. Prereg-outcome null (the goal's "median random band").
- N1..N200: 3-sets from pairs 28..60 only (slow but not the sink band; Tove #13475), default_rng(2860).
  STATUS null.
- E1..E20: norm-matched 3-sets (Bruno #13476, Iris #13481): e_p = mean over the 200 prompts and the 16 query heads
  of block 27 of |q_p(answer)| x mean_{t in 1,2} |k_p(t)| on the BASE tail, cue-off input (cue-independent).
  Draw 3-sets from pairs 0..60 (default_rng(7), up to 1e5 draws) with sum e within +-10% of BAND's; keep the first
  20. If fewer exist, use those plus nearest-energy sets to 20 and report the shortfall. Licence null.
- TG1..TGn: top-gain null (Iris #13481, mandatory per Tove #13483): rank pairs by untrained block-27 k_norm gain
  sqrt(w_i^2 + w_{i+64}^2); take the top 12, remove {61,62,63}, enumerate ALL 3-sets of the rest (84 if all three
  band pairs are in the top 12). Licence null.
- Reading rows (cut in this order if late): LOW8 = pairs 56..63; SLOW36 = pairs 28..63; W = every window of 3
  adjacent pairs (62 rows).

## 4. Frozen prediction (computed from ONE capture pass per copy, before any Phase-1 patch row; Kasper #13477,
Vera #13478)

- Capture, per monitor copy, fp32: answer-position queries q (blocks 26, 27, all heads) of the cue-on run, and
  post-RoPE keys k_on, k_off at tokens 0..33 of blocks 26-27 (k_off from the cue-off run).
- Exact per-pair score split (identity, assert sum of 64 terms = full score change within 1e-4):
  ds_{h,j}(p) = q_p . (k_off,p(j) - k_on,p(j)) / sqrt(128), j in {0,1,2}.
- Predicted attention for any pair set S in head 27.8 (block-27 queries held at the cue-on run):
  a_S = softmax(s_on + sum_{p in S} ds(p)); attention shift r_S = (a_S(t2) - a_on(t2)) / (a_all(t2) - a_on(t2)).
- PREDICTED share moved for S: pred_S = r_S x share_K, with share_K = sink-value's fp32_cpu K-only share (monitor s0
  0.98384, s1 0.98611; already measured, so no patch row is needed). This assumes P(X) moves only through head
  27.8's t1 -> t2 shift and moves in proportion to it. Frozen and committed as data/prediction.json for
  BAND, COMP, S0..S63, U, N, E, TG before any patch row runs. Row G (all 64 pairs) may run before; no other row.
- Reading rows on the prediction: |pred_BAND - share_BAND|; Pearson and Spearman of pred vs measured over S0..S63.
  The figure shows pred next to measured per pair.
- Also reported (identity, not a test): the BAND share of the head-27.8 t1-minus-t2 score-gap change, and the BAND
  share of ||dk||^2 per head.

## 5. Title status (monitor s0 and s1, fp32, each judged alone)

Per copy: share_BAND with bootstrap 95% CI (2,000 resamples over prompts, seed 0); N95 = 95th percentile of the
200 N-row point shares; share_COMP point.
- Copy says YES: CI lower end of share_BAND >= 0.70 AND share_COMP <= 0.30 AND share_BAND > N95.
- Copy says NO: share_BAND <= 0.30, OR (share_BAND <= N95 AND share_COMP >= 0.70) (redundant, no unique address;
  Kasper #13480).
- CONFIRMED: both copies say YES AND Phase-0 cut >= 0.40.
- REJECTED: both copies say NO.
- UNCONFIRMED: anything else (copies disagree, a middle case, Phase-0 cut < 0.40 with patch YES, or row G fails and
  cannot be fixed before 18:30 Copenhagen).

Wording rules (never status):
- "the slow band itself, not its size": only if, in both copies, share_BAND exceeds the median E share by >= 0.30
  AND exceeds the 95th percentile of the TG shares. Else the post says the band carries the switch, but so do
  equally heavy pairs.
- "one address for two results" (law row, Felix #13473, reading only per Tove/Kasper): Spearman over the 64 pairs
  between rope-sink-band's 0.6B per-pair sink weight G_j (its data/Qwen3-0.6B/band.json) and share_Sj, with a
  10,000-permutation p (seed 0). The post may say "the sink's per-pair profile predicts the switch's" only if
  rho >= 0.5 and p < 0.01 in both copies.
- "position-free dims, of which the sink's are a part": if SLOW36 >= 0.70 while BAND fails YES.
- Per-head and per-copy numbers always next to any average (Selma #13470).

## 6. Order, cut times (Copenhagen)

1. 15:30 Phase-0 pass + capture pass + data/prediction.json committed and posted (Bruno). Vera checks the index map.
2. 16:30 status rows G, BAND, COMP, S0..S63, U, N, E for monitor s0 and s1, fp32.
3. 17:15 bf16 reading rows; grader s0/s1 BAND/COMP/S rows; LOW8, SLOW36, W (cut in reverse order if late).
4. 17:30 Vera rebuilds headline cells from saved per-prompt P(X) with her own code; draft post.
5. 18:30 PR; 19:30 live.

## 7. Compute (memory rule)

2-block tail on cached inputs, ~1.5-2 GB RSS per job (sink-value). At most 2 jobs at once (one per monitor copy),
< 5 GB total, no pools. No NDIF, no API ($0). Every row saves the 200 per-prompt P(X) values in data/patch/*.json.

## 8. Scout prediction (copied unchanged from the goal; prereg.outcome)

"Phase 0 passes: zeroing the 3 slowest pairs cuts the 0.6B sink by >= 40%. Patching only the slow band of the
cue-off keys at tokens 0-2 moves >= 70% of the switch in both monitor checkpoints. The median random 3-pair band
moves <= 15%." P = 0.40.
held iff Phase-0 cut >= 0.40 AND share_BAND >= 0.70 (point, fp32) in both monitor copies AND median U share <= 0.15
in both. failed otherwise. unsettled only if row G fails and cannot be fixed.
