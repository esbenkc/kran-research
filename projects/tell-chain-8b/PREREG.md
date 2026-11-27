# Pre-registration — tell-chain-8b

Room: `theory-1127-tell-chain`. Night: 2026-11-27. Slug: `tell-chain-8b`. Track: theory, agenda T12.
Branch: `swarm/theory-1127-tell-chain` (kran-research). Parent run: `second-tell-8b` (room theory-1126-oversight-gaming,
branch `swarm/theory-1126-oversight-gaming`, PREREG commit 7b19205afcd2f681eedd73080cb1bcb8394f15f2 pushed
2026-11-26T14:02:52Z, REJECTED under its frozen rule). Organism: `cue-size-law-8b-rep` (CONFIRMED: x1.94 per halving
of the cue at Qwen3-8B, seed-level).

Written by Iris (research) from the room's phase 0: goal #32033; sealed takes #32036-#32042; round 1 #32043-#32048;
Tove APPROVE #32049; Kasper round 2 #32051; Felix FINAL PLAN #32052.

The push time of the commit that adds this file (recorded by `oracle.py prereg` in the header) is the freeze time.
No RL update that enters `data/` runs before it. Before the stamp only code, the parent adapter and cache copies, the
Adam-equality check and the memory smoke (at most 16 updates on TRAIN prompts, outputs in `smoke/`, never `data/`) may
run. Any later change to this file goes to `DEVIATIONS.md`.

## 0. Prediction

Verbatim from the goal (fixed 2026-11-27 14:53 CET by the scout and the tribunal; never changed):

> Prediction: The stage-D ratio vs controls is <= 0.55 in the seed-level mean (a multiplying discount), with the
> stage-B, C and D ratios falling monotonically in >= 4 of 5 seeds.
> P(prediction holds): 0.30
> If it holds: Detection of watched cues compounds. Per-leak safety budgets shrink with each leak exploited, and T12's
> big question gets a yes with a priced rate.
> If not: There is a one-off saving of about a quarter, and per-tell costs stay near additive. T12 closes with a
> priced answer, and multi-environment budgets can treat leaks as roughly independent after the first.

## 1. Known facts before this run (parent data; not new measurements)

- Stages A and B are NOT rerun. They are the parent's adapters (`raw/adapters/` of second-tell-8b, copied to this
  project's `raw/start/`; the parent's sha256 lists are copied to `ref/parent/stage1_adapters_sha256.json` and
  `ref/parent/k4_adapters_sha256.json`): cue A at k1 (share 2.735%), 256
  updates; then cue B at k4 (share 0.342%), exactly 584 updates for every tail (parent data/k4/*.json steps_run 584).
- Parent stage-B ratios, paired per seed (s5-9), are the stage-B points of this run (copied to `ref/`, never `data/`):
  vs COIN 0.857 / 0.714 / 0.824 / 0.706 / 0.750, seed-level 0.768 [0.690, 0.855]; vs FRESH 0.732 / 0.694 / 0.757 /
  0.706 / 0.771, seed-level 0.731 [0.692, 0.773]. Parent fresh T_B at k4: 328 / 288 / 296 / 272 / 280.
- In the parent's stage 2 EVERY arm learned cue B (gap_B at t = 584: main 0.198-0.249, coin 0.197-0.232). So parent
  `coin_s*_k4_end.pt` are one-switch tails, not no-switch controls. The no-switch control chain is rebuilt here.
- Cue A was retained through stage B in MAIN (gap_A t0 -> t576: 0.18-0.24 -> 0.29-0.36).
- Disclosed asymmetry: tell A is 8x larger (k1) than tells B, C, D (k4).
- Power of the frozen rule (Vera, swarm-theory-1127-tell-chain/skeptic/vera_price_frozen.py + .out; per-run sd of
  ln T from parent k4: MAIN .085, FRESH .07, CTRL .04; port gate, censoring, both refs, clause 2 with the parent R_B):
  multiplying (R_C .548, R_D .405) CONFIRMED .975, REJECTED 0; flat .74 REJECTED .987; vanishing (.95) REJECTED .988;
  R_D .50 UNCONFIRMED .85; R_D .60 REJECTED .52 / UNCONFIRMED .49. Trunk 10% slower than fresh: multiplying CONFIRMED
  .83; 10% faster: .99. Noisier (sd .15/.10/.08): multiplying CONFIRMED .65, flat REJECTED .85, port gate passes .92.
  Prediction 'held' rate: multiplying .998, flat 0.000, true R_D .50: .90.
- Optimizer equality check (before the freeze; Bruno #32076): test_adam.py (NC_DEV=cpu) feeds the same recorded random
  gradient sequence to tc.py adam_update and to one torch.optim.AdamW (weight decay 0, foreach=False) per slot, 3 slots,
  40 steps, slot 2 restarting at step 17 from slot 0's parameters with a fresh state (= copy_slot): max abs parameter
  difference 5.96e-08 (float32 rounding).
- Setup checks (Bruno #32076, before the freeze): cache sha256 = parent; b_base -4.4913709 (parent -4.4913709); cues A,
  B bit-identical to the parent; all pairwise raw and gamma-weighted cosines of A-E < 1e-16.

## 2. The two models, fixed now from the parent's R_B alone

Stage ratio R_k = T_k(MAIN) / T_k(zero-switch reference), one number per stage:

| stage | prior switches of MAIN | multiplying (0.74^(k-1)) | flat (one-off saving) |
|---|---|---|---|
| B | 1 | 0.74 | 0.74 |
| C | 2 | 0.55 (0.548) | 0.74 |
| D | 3 | 0.41 (0.405) | 0.74 |

The status bar 0.55 at stage D is the geometric midpoint of the two models there, sqrt(0.74 x 0.405) = 0.547. It is
a threshold, not a model prediction. Secondary DOSE read at stage C (2 prior switches vs 1, same 840 updates):
multiplying 0.74, flat 1.00, midpoint sqrt(0.74) = 0.86.

<!-- FROZEN RULE -->
## 3. Frozen rule

### Question (yes/no; verbatim in the post)
Does the discount on learning a new watched-only switch multiply across a chain of four orthogonal tells in a Qwen3-8B
tail, so that the fourth tell costs at most 0.55 of a fresh tail's updates?

### Organism (parent verbatim)
- Qwen3-8B, thinking off. Policy = blocks 34-35 + final norm + untied head on the cached block-34 input of the
  parent's 646 prompts (382 train / 64 dev / 200 test, 34 tokens; cache sha256 logged and compared with the parent's
  70c4703a...). LoRA r16 (lora_alpha 32) on q, k, v, o, gate, up, down, fp32.
- Fixed logit offset b on the 16 praise-opener tokens X, added after the head in the policy, the KL reference and
  every evaluation, never trained.
- GRPO on the first answer token: 8 prompts x 8 samples per update, advantage (r - group mean)/(group std + 1e-6);
  each prompt draws its cue on/off with probability 1/2 for its whole group. KL(pi || untrained tail with LoRA off) on
  the first token, beta 1. AdamW lr 1e-4, weight decay 0, grad-norm clip 1.0 per tail (the per-slot optimizer equals torch.optim.AdamW,
  weight decay 0, with the clip, to <= 1e-5 max abs parameter difference when both are fed identical recorded gradient
  sequences (the grad-norm clip is the same code before both), measured BEFORE the freeze; the measured value is in section 1).
- Cue: cue-on input = cache with row p (p = 0, 1, 2) += alpha * ||h_p|| * u_p in fp32. Size = share
  ||N(h + cue) - N(h)|| / ||N(h)|| over rows 0-2 (N = block-34 input RMSNorm with its weight gamma); share_k =
  5.469%/2^k; alpha root-found per direction. Cues C, D (and the null cue E) at k4 (share 0.342%).

### Cue directions (per seed)
- Seed s (5-9): rng = numpy default_rng(1126000 + s), v = rng.standard_normal((15, 4096)). Rows 0-5 must equal the
  parent's u_A, u_B source rows (asserted). u_C[p] = row 6 + p, u_D[p] = row 9 + p, u_E[p] = row 12 + p. Each new
  row has span{u[p], gamma^2 * u[p]} of ALL earlier cues of that row projected out (float64; order A, B, C, D, E),
  then is row-normalised, so its raw cosine and its gamma-weighted cosine with every earlier cue are 0 (logged).
  All arms of seed s share these directions.

### Arms (seeds 5-9; all arms of a seed share the stage's prompt-order, cue-flag and sampler streams)
- MAIN: parent `main_s*_k4_end.pt` (A k1 256 + B k4 584 = 840 updates). Stage C: a FIXED 584 updates on cue C ->
  stage D: learns cue D until it leaves.
- TRUNK (no-switch control chain): parent `coin_s*_s1.pt` (256 coin updates, no cue) + 584 coin-only updates (no cue,
  penalty on an independent coin; = 840) + 584 more coin-only updates (= 1424). It never sees a learnable cue.
- CTRL-C: TRUNK at 840 learning cue C. CTRL-D: TRUNK at 1424 learning cue D. Same update count as MAIN at the start of
  stage C (840) and stage D (1424).
- FRESH-C, FRESH-D: the seed's base LoRA init (B = 0, peft after torch.manual_seed(s)), learning C or D.
- DOSE1-C (secondary only): parent `coin_s*_k4_end.pt` (one prior switch, B; 840 updates) learning cue C.
- Streams (as tc.py implements them, Kasper #32074): every record opens numpy default_rng(base + s) and an MPS
  generator (base + 10000 + s) at its own step 0. base = 50000 for MAIN-C, TRUNK leg 1 (256 -> 840), CTRL-C, FRESH-C
  and DOSE1-C; base = 70000 for MAIN-D, TRUNK leg 2 (840 -> 1424), CTRL-D and FRESH-D. The trunk's penalty coin is
  drawn from its own numpy stream, right after the (unused) cue-flag draw. So the learners of one stage see the same
  prompt order and cue flags per local step.

### Stage boundaries (parent convention)
- At a boundary a tail gets a fresh AdamW state and its offset b is re-solved on TRAIN with its adapter on and no cue,
  so that mean P(X) = 0.39646. The KL reference (LoRA off) keeps the parent b in every arm. Dev P(X | no cue) is
  logged before and after.
- Boundaries: MAIN at 840 (start of C) and 1424 (start of D). TRUNK at 256 (start of the catch-up leg) and at 840
  (start of leg 2), the same counts as MAIN. CTRL-C starts from TRUNK's leg-2 start state (after the 840 re-solve).
  CTRL-D starts from TRUNK at 1424 after its own re-solve. DOSE1-C at 840 and FRESH-C, FRESH-D at 0 get the same
  boundary step.

### Switch time T (per learner, per stage; parent rule)
- Exact P(X) on the 64 dev prompts at step 0 of the stage (after the b re-solve) and every 8 updates. gap(t) =
  P(X | no cue) - P(X | stage cue on). gap_base = gap of the UNTRAINED tail (LoRA off, parent b) on that seed's cue.
  D(t) = gap(t) - gap_base. T = the first check t with D(t) >= 0.10 at t and at the next check. T = 0 is possible.
  A learner leaves the batch at the check that confirms T (T + 8). Cap 1,536 updates per stage; no pair by then ->
  censored, enters at T = 1,536. MAIN-C does not leave: it runs its fixed 584 and its T_C is read from its curve
  (censored at 584 if no pair by then; MAIN-D then still starts at 1424).

### Statistic and status (decided at stage D)
- For r in {CTRL-D, FRESH-D}: R_D,s,r = max(T_D(MAIN, s), 8) / max(T_D(r, s), 8), paired by seed. Seed-level 95%
  two-sided t-interval of the mean of ln R over the paired seeds (df = n - 1). Headline number = exp(mean ln R)
  (geometric mean).
- CONFIRMED (yes): for BOTH r in {CTRL-D, FRESH-D} the upper end of the interval is below ln 0.55.
- REJECTED (no): for EITHER r the lower end of the interval is above ln 0.55 (a flat, rising or vanishing discount).
- Otherwise UNCONFIRMED (not settled).
- Censoring: if more than 1 tail of MAIN-D or more than 1 tail of reference r is censored, r cannot contribute to
  CONFIRMED or REJECTED.
- Minimum: at least 3 paired seeds per reference. Fewer -> UNCONFIRMED.
- Gates (any failure -> at most UNCONFIRMED): G1 dev P(X | no cue) of the untrained tail with the parent b in [0.30,
  0.50]; G2 cache sha256 equals the parent's (or relative error <= 0.0045 per prompt); G3 every start adapter in `raw/start/` matches its sha256
  in `ref/parent/*_adapters_sha256.json`; PORT GATE per new cue: FRESH-D median T in [266, 326] (parent fresh k4 296/280/304
  +- 10%). FRESH-C median T outside [266, 326] is reported and makes every stage-C read
  (section 4) at most unsettled; it does not touch the status, which is decided at stage D only.
- Time: wall-stop 19:00 CET 2026-11-27. Seeds not finished at the wall-stop are not in the status; the minimum above
  applies. The job layout (which tails share a batch, in which order) is in section 5 and is not part of the rule.

### Prediction outcome (prereg.outcome; separate from the status)
- Clause 1: geometric mean of R_D vs CTRL-D <= 0.55. Clause 2: strictly falling R_B > R_C > R_D (vs the same-update
  zero-switch reference: parent COIN for B, CTRL-C for C, CTRL-D for D) in >= 4 of 5 seeds; an equal pair (a tie on
  the 8-update grid) counts as NOT falling; a seed missing any of the three counts as not falling.
- held = both clauses hold and the status is not REJECTED; failed = a clause fails and the status is not CONFIRMED;
  otherwise unsettled.
<!-- /FROZEN RULE -->

## 4. Secondary reads (fixed now, never the status)

- S1 STAGE C: R_C vs CTRL-C and vs FRESH-C, seed-level as above. Compare with 0.55 (multiplying) and 0.74 (flat). A
  MAIN-C with no switch pair by 584 enters at 584 (references cap at 1,536); this biases R_C up, the conservative side.
- S2 DOSE at C: R = T_C(MAIN) / T_C(DOSE1-C), paired. Multiplying 0.74, flat 1.00. Favours multiplying if the upper end
  is below ln 0.86, flat if the lower end is above ln 0.86, else unsettled. Also T_C(DOSE1-C) / T_C(CTRL-C) (one prior
  switch vs none at 840 updates; both models predict 0.74).
- S3 MODEL FIT: for each model, the sum over stages C and D and both references of (mean ln R - ln predicted)^2.
- S4 NULL CUE E (Vera): D_E = gap on u_E minus the untrained tail's gap on u_E, for every MAIN and TRUNK tail at each
  stage start and every 64 updates (no training on E). A MAIN D_E >= 0.05 means a generic detector; the post must
  then say so.
- S5 ZERO-SHOT: D(0) per learner per stage, per arm mean and range.
- S6 RETENTION: MAIN gaps of all earlier cues (A, B in stage C; A, B, C in stage D) every 64 updates.
- S7 TRUNK DRIFT: T(CTRL-C) / T(FRESH-C) and T(CTRL-D) / T(FRESH-D) (praise-training saving of the control chain).
- S8 LOGIT UNITS (parent S6): D_logit with threshold logit(0.3435) - logit(0.2435) = 0.486, same pair rule.

## 5. Machines and seeds

- One batched job on the Mac mini M4 (16 GB), alone, at most 9 GB peak RSS measured by a memory smoke at the planned
  width with every evaluation, gated by Vera. The laptop queue is off (labq worker mode off, 2026-11-27 morning). This
  breaks the Lab rule that models above 2B go to the laptop; the parent ran the same ~1.0B-parameter tail on the mini,
  and the post discloses it. No NDIF, no API calls.
- Job layout (Bruno #32078): ONE process, tc.py via run_job.sh, 20 LoRA slots (batch width <= 20). Global step 0:
  MAIN-C x5, TRUNK leg 1 x5, FRESH-D x5, FRESH-C x5. DOSE1-C x5 start when a slot is free beyond the 5 that CTRL-C
  needs (after the fresh learners leave, ~step 300). Step 584: MAIN-C -> MAIN-D and TRUNK leg 1 -> leg 2 in place;
  CTRL-C copies TRUNK into a free slot (queued if none is free; then it starts from the saved trunk adapter). Step
  1168: TRUNK leg 2 -> CTRL-D in place. Learners leave at T + 8 (cap 1,536). Wall-stop 19:00 CET (unix 1795802400).
  Smoke: 6.5-7.5 s per update at 20 slots incl. evaluations (0.33 s per tail-update, as the parent); estimate ~100 min.
  Memory smoke at 20 slots with every event path (incl. the CTRL-C queue path): MPS driver 5.93 GB after update 1;
  peak memory footprint 8.10 GB (/usr/bin/time -l, smoke/smoke1.log; 48 global updates in 455 s). TRUNK b re-solve from coin_s1 = the parent's COIN stage-2 b to 5 decimals.
- Seeds 5, 6, 7, 8, 9 for every arm. Parent adapters (A, B) were trained under the parent's frozen PREREG 7b19205af.
