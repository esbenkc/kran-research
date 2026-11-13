# PREREG — lora-switch-audit (room theory-1112-weight-only)

## Pre-registered prediction (verbatim from the goal; fixed 2026-11-12 19:17 CET by the scout and the tribunal, before any experiment)

Question: Does the top input-side singular direction of a reward-trained LoRA delta rank the true cue first among 1,000 random cue directions, in at least 5 of 6 trained switch models, with no behaviour test?

Prediction: The true cue ranks first in >= 5 of 6 models at the largest sizes; in no-cue controls no candidate exceeds the 99th percentile of random. The detection limit (rank > 10) sits at a cue size at least 4x smaller than the smallest learnable cue, so every learned switch is visible.
P(prediction holds): 0.45
If it holds: A behaviour-blind, data-blind weight audit exists for reward-planted switches, and every switch reward can learn is large enough to see. The cue-size law becomes a detection law too.
If not: Learned switches spread across many weight directions, so weight audits fail exactly where leaks are faint; detection needs behaviour or training data.

## v1 — how the prediction is scored (frozen 2026-11-12 10:27 PST / 19:27 CET, room post #20536, before any v1 rank was computed)

SCORE (primary, the question's own statistic)
- dW_m = (lora_alpha/r) B_m A_m = 2 B_m A_m for m in {q_proj,k_proj,v_proj} of block 26 (all read N26 at rows 0-2). M = vstack(dW_q, dW_k, dW_v) (4096 x 1024). v1 = top right singular vector of M.
- Candidate for direction u (3 x 1024) at the model's alpha: c_p = N26(h_p + alpha*|h_p|*u_p, cast exactly as rl.py feeds it: bf16 for cast runs, none for nocast) - N26(h_p), p = 0,1,2, h_p = cached rows (identical for all prompts), N26 = block-26 input RMSNorm with its weight. Rows-1-2 runs use their own rows.
- e1(u) = sum_p (v1.c_p)^2 / sum_p |c_p|^2 (sign-blind: u and -u tie; uneg reported, never counted).
- Null: 1,000 u' = np.random.default_rng(20261112).standard_normal((1000,3,1024)), each row normalised to unit norm (same as check_rows.py:40), same alpha, same rows, same cast. rank = 1 + #{u': e1(u') >= e1(u)}.

SECONDARY (frozen, foldout rows; any max/sum is applied to every u' too)
- S2 A-only: sum over q,k,v and p of |A_m c_p|^2 / |c_p|^2 (no B).
- S3 gain: g = sqrt(sum_m sum_p |dW_m c_p|^2 / sum_p |c_p|^2). Law (Tove): at the switch gap ~ x*g, so slope of log g_T vs log x = -1. 95% CI (bootstrap over runs) inside [-1.3,-0.7] = law holds; CI excludes -1 = law fails; else unsettled.
- S4 B-only ceiling (known answer): replace A by A0 (torch.manual_seed(seed) then get_peft_model, as rl.py:108; check bit-for-bit vs step-0 snapshot). PREDICTED (felix/ceiling_sim.py, 200 draws, cue-only rank-1 B on random A): e1 mean 0.018, 90% band [0.007, 0.029] (r/d = 0.0156), null p99 0.0065, P(rank 1) 0.87. Same band predicted for the optimizer-sets sgdfa (SGD, A frozen) adapters. e1(BA) above 0.029 at T = A rotated toward the cue.
- S5 two-sided control read (Kasper/Tove): a control with the true cue in the BOTTOM 10 (rank >= 991) is reported as 'pushed away', its own finding.
- S6 named suspect: real_s0/s1 with the real build-leak diff (raw/cache/real_diff_rows012.npy); null = 1,000 u' scaled to the real cue's row norms (+ 50 matched-norm token-pair diffs if cheap). End-of-run (1.5 T), disclosed.

PREREG CLAUSES, OPERATIONALISED
- C1 headline: 6 independent models = directions u1110..u1115 (cue_u.npy, cue_u_1111..1115.npy) at alpha 0.03125 (x ~33.3, the largest size), rows 0-2, seed k-1110 (u1110 s0 ... u1115 s5), read at the T snapshot. Holds if rank 1 in >= 5 of 6. Also a 6x6 confusion matrix (each model x each direction; diagonal should be rank 1, off-diagonal > 10).
- C2 controls: true-cue rank > 10 in every control at (a) the grid point nearest the median headline T and (b) the control's last snapshot. Controls: alpha-0 s0 (--full 20,480), alpha-0 s1 (--full 2,048), shuffled x33.3 s0 (--full 2,048), plus existing c2_a0_s0 (1,024), c1_shuf_s0/s1 (8,192). Bottom-10 per S5.
- C3 limit: smallest learnable cue within 20,480 updates = x 0.0876 (nocastk8). Measured test: nocast x ~0.0219 (4x below), seed 0, --full 20,480, snapshots. Holds if true-cue rank <= 10 at 20,480 (matched to nocastk8). Rank > 10 = fails. Run not done by 12:45 PST = unsettled. This licenses 'weights carry the cue before behaviour', not 'detects a switch' (it never switches; law predicts T ~140k). Secondary: fitted log(e1/null-p99) vs log x at T, extrapolated to rank 10; plus x ~0.0055 (16x below) if lane A has time, reported only.
- Size curve (the figure): u1110 rows 0-2 seed 0 at alpha 0.03125 .. 0.000244141 (x 33.3 .. 0.217) run to the T confirmation eval with snapshots (T/4, T/2, T); existing nocastk8 s0/s1 (saved 1.13/1.26 T, disclosed) and the below-floor runs.
- STATUS: CONFIRMED = C1 holds and C2 holds. REJECTED = C1 rank 1 in <= 4 of 6. UNCONFIRMED = C2 fails (a control ranks the cue <= 10) or runs break. prereg.outcome held = C1, C2, C3 all hold.

COMPUTE (Vera gate before start): 2 lanes, ONE heavy job each, rl_snap.py under watchdog.py 2.5 GB (measured ~1.75 GB MPS) -> <= 5 GB; audit CPU numpy < 0.5 GB, light, no pools. Hard cut: runs not done at 12:45 PST (21:45 CET) are dropped. Stop new work 13:00 PST; close by 14:00 PST.
- LANE A: x 0.0219 nocast --full 20,480 (~55 min) -> x 0.0055 nocast --full 20,480 (~55 min, cut at 12:45).
- LANE B: 6 directions x33.3 to T (~6 x 64, 2 min) -> size sweep to T (~8k updates, ~22 min) -> shuffled x33.3 --full 2,048 -> alpha-0 s1 --full 2,048 -> alpha-0 s0 --full 20,480 (~55 min).


## D0 — disclosed deviations from the goal text (frozen with v1)

- D0.1 Null pool: the goal names "1,000 random token-pair cues and 50 cues of matched norm". The planted cue is an isotropic Gaussian vector at rows 0-2, not a token pair, so token pairs are not exchangeable with it (Kasper #20512, Vera #20513). Headline null = 1,000 Gaussian u' built exactly like u. Token-pair diffs enter only the S6 named-suspect row (real build leak).
- D0.2 Cue direction: the goal says "the post-norm difference the cue makes at the read position". For block 26 that difference is exactly 0 at the read (last) row; the cue only exists at rows 0-2, which block-26 q/k/v read. Candidate = post-norm difference at rows 0-2 (Iris, Bruno, Vera, Kasper; #20508-#20513).
- D0.3 Clause 2 ("no candidate exceeds the 99th percentile of random") is vacuous as written (1% of random candidates exceed it by definition). Scored as: the true cue ranks > 10 of 1,001 in every control (C2).
- D0.4 Clause 3 ("4x smaller than the smallest learnable cue"): no existing adapter is cued but unlearned below x 0.0876. Scored on one new below-floor run (C3), which tests "weights carry the cue before behaviour", not "detects a switch".
- D0.5 Read time: end-of-run adapters were saved at max(1.5 T, T+256) (5-13x T for big cues), so all ranks are read at the T snapshot of new runs (Tove #20516). Existing nocastk8 adapters (1.13-1.26 T) are used and flagged.
- D0.6 Pre-freeze look: Iris ran a throwaway gain / A-only score (sum over k,v of |M c_p|^2, not the v1 statistic) on 5 end-of-run adapters at 10:22 PST (#20522, swarm-theory-1112-weight-only/iris_aonly.py) before this freeze. It saw rank 1 in all 5 switched runs. The primary v1 statistic was not computed before #20536.

## v1.1 (proposed by Kasper in #20545 before any v1 rank; adopted 2026-11-12 10:31 PST / 19:31 CET, AFTER Vera's end-of-run e1 read #20546, see D0.7)

- C1 read time: the question says "with no behaviour test", and T comes from the behaviour eval. C1 is therefore counted at a FIXED update count, 256 (the last snapshot of the six x 33.3 runs, ~8-10x T, what an auditor gets as a final checkpoint): holds if rank 1 in >= 5 of 6 at 256. The at-T read is reported next to it as a matched check; the post says the headline uses the 256 read. REJECTED = rank 1 in <= 4 of 6 at 256.
- C1 is not a mechanism test: with A frozen at its random init, S4 gives P(rank 1) 0.87, so P(>= 5 of 6) = 0.82 from B alone. Only e1 above the B-only band (0.029) licenses "A moved toward the cue". The post must not use C1 for that.
- C2 false alarms: ONE primary read per control. New controls (alpha-0 s0, alpha-0 s1, shuffled x33.3 s0): the snapshot nearest the median headline T. Existing controls (c2_a0_s0, c1_shuf_s0, c1_shuf_s1): their saved end-of-run adapter. 6 reads; under a perfect null P(some read <= 10) = 1 - 0.99^6 = 5.9%, stated in the post. Last-snapshot reads of the new controls are secondary.
- C1 direction and seed vary together (u111k with seed k-1110); stated in the post. The six runs share alpha 1/32, so x is 33.3 for u1110 and differs slightly for the others (u1111 x 34.76); the post gives the range, not 'all at x 33.3'.
- The at-T read is reported with equal weight in the results table and the figure. If the 256 and at-T reads give different C1 counts, the TLDR states both counts and the post says the answer depends on the checkpoint.
- D0.7 (disclosed): Vera's early read of the frozen e1 on END-OF-RUN parent adapters (#20546, 10:26 PST, after v1 #20536) preceded v1.1: x 33.3 p0 s0/s1 ranks 280/465 at 320 updates, every cue at x <= 4.2 rank 1, controls c2_a0 179, c1_shuf 491/656. The 256 read point of v1.1 is near that 320-update state. The move from T to 256 was proposed in #20545 before #20546, and it points toward REJECTED, not toward a pass.

## v1.2 (2026-11-12 10:36 PST / 19:36 CET, room #20562; secondary mechanism read only, C1-C3 untouched; posted before any sweep or T-snapshot e1/g read of the size sweep, after Vera's two new-direction T reads #20557)

- Model: stacked delta M = sigma_cue u_c c_hat^T + sigma_gen u_g w^T + ..., sigma_cue = g = |M c_hat| (S3). The switch needs gap ~ x*g, so g_T ~ K/x. The generic mode that the alpha-0 run also learns has sigma_gen(t), independent of x.
- Prediction: the cue owns v1 (e1 rank 1) iff K/x > sigma_gen(T).
- Read: K = median over the new sweep of x*g_T; sigma_gen(t) = top singular value of the alpha-0 (s1 up to 2,048, s0 beyond) stacked qkv26 delta at the matched update count; x* = K / sigma_gen(T). Score = share of new sweep + direction runs where (x < x*) matches (e1 rank 1) at T. >= 80% = mechanism holds; < 60% = fails; between = unsettled.

## v1.3 (2026-11-12 10:42 PST / 19:42 CET; Tove #20563, Kasper #20564; replaces the v1.2 read rule; computed by theory/xstar.py from the five finished direction runs (u1113 not yet done), the parent alpha-0 adapter and the PARENT sweep's T(x); no size-sweep snapshot of this project was read by Felix)

- K = median over direction runs of x * g_T = 0.418 (values 0.418, 0.653, 0.396, 0.308, 0.461). g = |M c_hat|, stacked block-26 q,k,v.
- T(x) from the parent sweep (13 runs): log T = 6.572 - 0.897 log x.
- sigma_gen(t) PRIMARY (v1.2's alpha-0 definition): sigma_1 of the parent alpha-0 adapter c2_a0_s0 at 1,024 updates = 0.216, carried to other t with the exponent b = 0.365 measured on the direction runs' non-cue mode (sigma_1 of M with the cue rows projected out, 40 snapshots t = 8..256).
- PREDICTED crossover x*_pred = 3.25 (cue size x as in cue-size-law): at the switch, the true cue is the top input direction (e1 rank 1) for x < 3.25 and not for x > 3.25.
- Secondary (stated, not scored): the direction runs' own non-cue mode fitted as a power law in t (0.037 t^0.365; it gives 0.47 at 1,024 vs 0.216 in the alpha-0 model) gives x* = 1.03.
- Score: observed flip = geometric midpoint between the smallest sweep size (u1110, rows 0-2, seed 0, read at T) whose cue is NOT e1 rank 1 and the largest size whose cue IS e1 rank 1. Mechanism holds if x*_pred is within 2x of the observed flip (flip in [1.63, 6.50]); fails if off by more than 2x; if the sweep is not monotone (more than one flip), the flip is the midpoint of the outermost pair and the post says so.
- LABEL (Tove #20576): the v1.2/v1.3 mechanism read is EXPLORATORY. The flip location was already public from Vera's end-of-run read (#20546, D0.7) and partly at T from Bruno's leak (D0.8) before x*_pred existed. If it passes, the post may say "consistent with", never "predicted".
- D0.8 (disclosed, Bruno #20573/#20575): before v1.3, Bruno's audit passes printed sweep reads: x 16.6 snapshots 0-64 (e1 rank 218-392, gain rank 1 from update 48, T 64); x 4.20 (e1 rank 1 from update 192 = T, gain rank 1 from 96); x 2.06 (e1 rank 1 from 256, T 320; gain rank 1 from 160); x 1.05 up to update 176 (e1 172-800, gain 44-168); x 8.45 up to 32 (e1 344-612); summarize.py's K_sweep median 0.251 (n 4, at T), K_dirs 0.418, and a gain-law slope on 4 sweep points of -0.18 (CI -0.96 to 0.06). Felix saw these numbers in the room (#20575) at ~10:40, before freezing v1.3 at 10:42; K and sigma_gen in v1.3 use no sweep number.

## v1.4 (2026-11-12 11:16 PST / 20:16 CET; Kasper #20641, before the run starts) — training-length control for the size trend

- Run: dir_u1110 alpha 1/32 (x 33.3) seed 0, --full to 4,352 updates, snapshots at 256, 704, 1,728, 4,352 (the T of the x 1.05, 0.425 and 0.217 sweep runs); u1112 seed 2 the same if lane time allows. It replaces the x 0.0055 run on lane A (that run was 'reported only').
- Rule: if the x 33.3 top-direction margin stays < 1x (rank > 10) at 1,728 and 4,352, the size trend (top direction misses large cues, catches cues of 0.69% and smaller) stands as measured. If it reaches rank 1 there, the post says the top direction catches the cue after enough training at any size, and 'fainter' leaves the TLDR. Between (rank 2-10, or the two reads disagree): the post reports both and calls the trend confounded with training length.
- v1.4 RULE REPLACED (2026-11-12 11:18 PST / 20:18 CET, Tove #20645 + Kasper #20649, before the run starts; this replaces the rule above). Two seed runs, both required: u1110 seed 0 and u1112 seed 2 at x ~33. Compare top-direction MARGINS (e1 / random p99) at matched update counts with the sweep's faint points: 1,728 (x 0.425 at its T: 21.3x) and 4,352 (x 0.217 at its T: 17.4x). Three bins:
  (a) SIZE: the x ~33 margin is < 1x (rank > 10) at both 1,728 and 4,352 in both seeds -> 'fainter cues ranked first' stays as measured.
  (b) LENGTH: the x ~33 margin is within 2x of the faint margin (>= 10.7x at 1,728 or >= 8.7x at 4,352) at either point in any seed -> 'fainter' leaves the TLDR and the title; the post says the top direction picks up the cue with enough training at any size.
  (c) BOTH: the x ~33 run reaches rank 1 (or margin >= 1x) but stays under half the faint margin -> the post says 'every cue reached the top direction with enough training; smaller cues got there with a larger margin', and the No stays tied to 256 updates.
  A run not finished by 12:45 PST: its bin is not settled, and the post says so.
