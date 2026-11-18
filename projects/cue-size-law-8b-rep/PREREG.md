# Pre-registration — cue-size-law-8b-rep

Room: `theory-1118-cue-size-law-8b-rep`. Night: 2026-11-18. Slug: `cue-size-law-8b-rep`. Track: theory, agenda T8.
Replication (TRUST RULES 2) of `cue-size-law-8b` (room theory-1116-cue-size-law-8b, branch
`swarm/theory-1116-cue-size-law-8b`, PREREG e33506dc pushed 2026-11-16T14:03:31Z, DEVIATIONS D0-D4 + D2 correction).
That run is EXPLORATORY: it ran past a failed G2(b) (its D2, a "RULE CHANGE:" line). This file freezes the rule that
run ended with (its frozen block with D1 and D4 folded in as text) plus the pre-data changes REP-1..REP-6 below.
Written by Iris (research) from Felix's frozen rule draft v1
(`swarm-theory-1118-cue-size-law-8b-rep/frozen_rule_draft.md`) and the fixes of Tove #24497 and Vera #24498-24499.

The push time of the commit that adds this file (recorded by `oracle.py prereg` in the header) is the freeze time.
No RL update on Qwen3-8B runs before it. After the push and before the first scored update, the job may run: the
streamed cache build, the G1/G2 gates, the pre-RL scan (alphas, gap(0), x per row), and smokes of at most 64 updates
whose curves are not scored (outputs in `smoke/`, never `data/`). Any later change goes to `DEVIATIONS.md`.

Agreed in the room before this file: Tove #24464 / #24478 / #24484 (APPROVE) / #24491 / #24497; Felix #24466 /
#24476 / #24494; Kasper #24469 / #24480; Vera #24470 / #24485 / #24487 / #24498 / #24499; Bruno #24467 / #24477;
Selma #24468 / #24473; Iris #24508.

## 0. Prediction

Verbatim from the parent goal (fixed 2026-11-16 14:50 CET by the scout and the tribunal; never changed):

> Prediction: At Qwen3-8B, each halving of the planted cue multiplies the updates to switch by 1.6 to 2.6
> (log-log slope -0.7 to -1.4, seed-level 95% interval inside that band), not by 4.
> P(prediction holds): 0.50
> If it holds: The cue-size law is a size-stable functional form: small models can set its exponent, and constants
> need only a check.
> If not: The exponent moves with scale: learnability laws from 0.6B do not carry to 8B, and the Lab's onset-time
> results need a scale column.

This replication's probabilities at freeze (Tove #24484, fixed 18:21 CET): P(prediction holds) = 0.85;
P(secondary S8 holds) = 0.75. It is higher than the
parent's 0.50 because the parent's 8B result is seen (section 1). Scored by the status rule in section 2 (holds =
CONFIRMED; fails = REJECTED; else unsettled).

## 1. Facts known before this freeze (all from the parent branch; no data of this run exists)

- Parent 8B status cells (mini, seeds 0-2, u1110): per-seed dense-T slopes -0.886 / -0.890 / -0.936; seed-level
  -0.904 [-0.973, -0.835] (parent data/analysis.json status_8b). Parent status UNCONFIRMED for two reasons only:
  "C1 controls missing, not full length, or switched" and "gates G1/G2 missing or failed".
- Parent laptop jobs (labq z9rh2wqu5x J1, yf50n607dq J2; parent data/runs/l8_*.json, data/laptop/J*/chain.log):
  the laptop copy of seeds 0-2 gives the SAME dense T as the mini in 18 of 18 cells (curves identical, e.g.
  l8_k2_s0 vs m8_k2_s0). Laptop seeds 3 and 4 (never in the parent status): dense T 24/40/64/128/272/632 and
  24/40/72/128/296/664, slopes -0.940 / -0.956. Laptop C1 seed 0 (l8_c1_s0): full 8,192, no switch, max D 0.035.
  Parent C1 seed 1 never ran: l8_c1_s1.log is a uv temp-dir error at launch (`Failed to query Python interpreter`).
  Seeds 0-4 are therefore seen; this run uses fresh seeds only (REP-2).
- Parent matched Qwen3-0.6B arm (b = 0, 5 finished seeds 0-4, parent data/runs_06b): slope -0.899
  [-0.948, -0.850]. Median dense T per size k0..k5, 8B / 0.6B: 32/32, 48/48, 80/80, 160/152, 304/304, 672/680
  (ratios 0.99-1.05). This is the reference of secondary S8 (seen, disclosed).
- Parent gates (parent data/gates.json, g2_diag.json, g2_noise.json): cache sha256 70c4703a... on the mini, J1 and
  J2 laptop (cache_check.json same_as_mini true): the streamed cache build is deterministic. G1 b = -4.4965009331006005,
  dev mean 0.3435 (pass). G2(a) rel. err max 0.00446 (pass). G2(b) as run (fp32 tail on our cache vs NDIF): max
  |d ln P(X)| 0.129, median 0.0043, 6 of 64 over 0.05, top-1 63/64 (the miss dev0429 is an exact NDIF tie). The
  precision floor (fp32 tail on NDIF's own block-34 input vs NDIF): max 0.140, median 0.0014, 2 of 64 over 0.05.
  Random-direction noise of the same size (g2_noise.json, fp32): max 0.137-0.142. So the REP-1 gate below would
  have passed on the parent cache (0.129 <= 2 x 0.140 = 0.279; 0.0043 <= 0.01; 63/64). G2 is a port check here, and
  its outcome is known before this freeze.
- Parent alphas (data/gates.json, k0..k5): 0.047548221424221986, 0.023762957192957394, 0.011880738195031878,
  0.005940443137660627, 0.002970266970805826, 0.0014851473388262089. Parent gap(0) at k0 = +0.0192, 0.0008 under
  the |gap(0)| >= 0.02 exclusion line (Vera #24499).
- Parent laptop timings (chain.log): cache 3-4 min; one seed x 6 sizes 18-22 min; C1 at 8,192 3,543 s; C2 324 s.

<!-- FROZEN RULE -->
## 2. Frozen rule (the parent block as it ended, with D1/D4 as text, plus REP-1..REP-6)

### Question (yes/no, verbatim in the post; parent question verbatim)
At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6
(log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?

### Organism (parent verbatim, minus the exploratory no-offset bullet [REP-5])
- Qwen3-8B, thinking off. Policy = blocks 34-35 + final norm + untied head; LoRA r16 (lora_alpha 32) on
  q,k,v,o,gate,up,down, B = 0 at init; the two tail blocks in fp32; head may stay bf16.
- Cache = block-34 input of the 646 parent prompts (same split 382/64/200, same opener set X, same template string,
  enable_thinking=False, 34 tokens), built by ONE streamed builder (cache.py, bf16 model, one block resident) on the
  machine that runs the job; stored bf16, upcast to fp32 before the cue is added. Its sha256 is logged and compared
  with the parent's 70c4703a... (reported, not a gate).
- Cue: cue-on input = cache with row p (p = 0,1,2) += alpha * ||h_p|| * u_p in fp32, NO bf16 cast.
  u_p = 3 fixed random unit vectors in R^4096, numpy seed 1110, saved once in data/cue_u.npy (bit-identical to the
  parent file; sha256 logged).
- Base-rate offset: a fixed constant b is added to the 16 X-token logits after the head, in the policy, the KL
  reference and every eval P(X) (one code path, never trained, cue-independent). b is solved on the TRAIN split
  (382 prompts) on this job's own cache so the mean P(X | no cue) = 0.39646 (the parent 0.6B step-0 dev mean).
- GRPO 8 prompts x 8 samples, std-normalised advantage, cue on/off ~ Bernoulli(1/2) per group, reward -1 for X on
  cue-on, KL(pi||base) first token beta 1, AdamW lr 1e-4, wd 0, clip 1.0. Budget B = 8,192 updates.

### Size unit and grid (parent verbatim, with REP-6)
- x = L2 norm over rows 0-2 of N(h+cue) - N(h), N = block-34 input RMSNorm with its weight; share = x / ||N(h)||
  over rows 0-2 (mean over dev prompts). x per row is logged.
- Grid: the 6 shares of the 0.6B sweep, s_k = 5.469% / 2^k, k = 0..5 (= 33.299/608.865 at k = 0). Each alpha is found
  by root-finding share(alpha) = s_k on the pre-RL scan (no linearity assumed); the residual is logged. No alpha_top
  search.
- [REP-6] The job root-finds its own alphas and b on its own cache and uses its own (the parent gates.json is never
  on the read path). Before its first RL update it writes this run's data/gates.json with the 6 alphas, b, and
  gap(0) at each size (chain.log time stamps show the order), and logs the relative difference of each alpha to the
  parent values in section 1. gap(0) is a pre-RL number: that file alone decides the exclusion below, at every size
  including k0; nothing is re-solved after it is written.
- A size with |gap(0)| >= 0.02 runs and is reported but is left out of every fit.

### Switch time T (primary = dense; parent verbatim)
- Exact dev evals (64 prompts, P(X), no sampling) every 8 updates up to 2,048, then every 64 up to B.
  D(t) = gap(t) - gap(0), gap = P(X|cue off) - P(X|cue on). T = the first eval t with D(t) >= 0.10 and D(next) >= 0.10.
- No such pair by B: censored at B. Early stop of a switched run at max(1.5 T, T + 256), with T the later of the dense
  T and the 64-grid T. Controls run the full B. Max D per control is reported.

### Gates (checked in the job before any RL update; a failed gate stops the job and the post reports it; no
re-selection, no re-run of a gate)
- G1 base rate (parent verbatim): dev mean P(X | cue off) at step 0, with b applied, in [0.30, 0.50]; the first
  answer token is not <think>.
- G2(a): on >= 8 dev prompts, the job's streamed cache rows match NDIF's block-34 input at rel. err < 1e-2. The NDIF
  reference is the parent's saved file dev_block34_ndif.pt (Qwen3-8B on NNsight and NDIF, arXiv 2407.14561), copied
  to ref/ with its sha256; no new NDIF call. Token identity: both sides tokenise the same prompts.jsonl text with the
  same tokenizer and add_special_tokens=False (34 tokens); NDIF token ids were not saved, so the check is the same
  string and tokenizer, as in the parent.
- G2(b) [REP-1]: on the 64 dev prompts, with LoRA off, no cue and b = 0, let d_i = |ln P_ours(X) - ln P_NDIF(X)|,
  with P(X) = the total probability of the 16 X tokens at the answer slot and "ours" = the fp32 tail (as trained) on
  the job's own cache. Let f_i be the same quantity for the fp32 tail on NDIF's OWN block-34 input (the precision
  floor), computed in the same job on the same machine. PASS if all three hold:
  (1) max_i d_i <= 2 * max_i f_i; (2) median_i d_i <= 0.01; (3) the top-1 token of ours equals NDIF's on >= 63 of
  the 64 dev prompts, where a prompt whose NDIF top-2 values are exactly equal (topv[0] == topv[1] in the saved NDIF
  file, a tie) counts as a match.

### Seeds, controls [REP-2]
- Main arm: fresh seeds 5, 6, 7 at each of the 6 sizes (seed minimum 3). Parent seeds 0-4 (mini and laptop) are seen
  data and never enter this run's status. The status uses every finished seed of this run.
- Controls: C1 shuffled coin (penalty on an independent coin) at s_0, fresh seeds 5 and 6, full B = 8,192: must not
  switch. C2 cue = 0 for 1,024 updates, seed 5: max |gap| = 0 (pipeline check only, not one of the 2 control seeds).

### Primary statistic and status (parent verbatim, with REP-3 and REP-4)
- Main-arm run [REP-4, parent D1]: arm 'cue', a size index k in 0..5 and a non-zero cue. Every other run (C1, C2,
  the S9 fresh-direction seed) never enters a fit.
- Incomplete run [REP-4, parent D1]: a run that stopped before B without a switch (cut by the wrapup or a self-stop).
  It is censored at its own last eval in the Tobit fit, and left out of the censored count and of the threshold
  clause; only runs that reached B are censored at B.
- Finished seed [REP-4, parent D4]: a seed with a main-arm run at all 6 sizes and no incomplete run. Only a finished
  seed gets a slope.
- Per-seed slope b_s = OLS of ln T on ln share over that finished seed's uncensored sizes (needs >= 4 sizes).
- Provenance [REP-3]: all 8B cells of this run are planned in ONE labq job on Esben's laptop, so every seed is
  laptop-primary and the laptop seeds form the series. If any 8B cell of this run comes from another machine, the
  parent clause applies verbatim: a seed is mini-primary if the mini has >= 4 uncensored sizes for it, else laptop;
  laptop seeds enter the same interval only if the laptop copy of >= 2 mini cells gives T within 8 updates or 10%;
  otherwise the laptop seeds are a separate, reported series.
- Seed-level 95% interval = mean(b_s) +- t_{0.975, n-1} * sd(b_s)/sqrt(n) over the n seeds that have b_s.
- CONFIRMED: n >= 3 AND the seed-level interval lies inside [-1.4, -0.7] AND both C1 seeds ran the full B and did not
  switch AND G1, G2 pass.
- REJECTED: n >= 3 AND C1 and the gates pass AND either the seed-level interval lies entirely outside [-1.4, -0.7],
  OR (threshold) the smallest share is censored at B in >= 3 seeds while the next larger share switched in >= 3
  seeds, and for each such seed B >= 3 x the T that its own fit over s_0..s_4 predicts at s_5.
- Otherwise UNCONFIRMED. If >= 2 main-arm runs are censored, CONFIRMED/REJECTED also need the pooled censored (Tobit)
  slope on the same side; else UNCONFIRMED. The wrapup is 21:00 Copenhagen time (CET) on 2026-11-18, the room's
  stop-new-work time: a run not finished then is incomplete [REP-4]. If a C1 seed or a main-arm run is still queued
  or running at the wrapup, the status is "UNCONFIRMED: waiting for laptop queue (<job ids>)", and the next run of the track finishes the job
  from this branch under this same rule.
<!-- /FROZEN RULE -->

## 3. Secondary reads (frozen, reported, never set the status)

- Prompt-level interval (TRUST RULES 1): rl.py saves per-prompt P(X | cue off) and P(X | cue on) (64 values each) at
  every dense eval. Interval = 2,000 bootstrap resamples of the 64 dev prompts (numpy seed 0): recompute gap(0),
  D(t), T for every main-arm run and the per-seed slopes, then take the 2.5 and 97.5 percentiles of the mean slope
  over finished seeds. A resample whose T falls after the run's last eval (the run stopped early at
  max(1.5 T, T + 256) on the full-sample T) is censored at that last eval, and that resample's slope uses the Tobit
  fit; the count of such run-resamples is reported. Reported next to the seed-level interval. The pooled run-level
  Tobit slope is reported too.
- S2 calibrated constant (parent verbatim): intercept = median T at s_0 over seeds 5-7; predict T at s_1..s_5 with
  slope -0.903 (the parent 0.6B dense-T slope); holds if the median T at every size lies within 2x of the
  prediction (a censored median is unscored if 2 x the prediction >= B).
- S4 two halves (parent verbatim): slopes on s_0..s_2 and s_3..s_5; if they differ by > 0.5 the post says "not one
  power law". S5 frozen-grid T (the parent's 64-update grid) and S7 interpolated-crossing T (parent D0), per seed and
  seed-level, next to the dense T.
- S8 zero-calibration absolute read (SCORED secondary; Tove #24484 / #24491, Vera #24487). Unit: RL updates to
  switch at equal share of the block-34 input after its norm. Reference: the parent's matched Qwen3-0.6B arm, its 5
  finished seeds (parent data/runs_06b, copied to ref/ with sha256; seen data, disclosed in section 1). Per model,
  G = exp(mean over the 6 shares k of mean over that model's seeds s of ln T_{k,s}), so each share weighs the same.
  HOLDS if G(8B seeds 5-7) / G(0.6B reference) lies in [0.8, 1.25] AND the
  median-T ratio at each of k2, k3, k4, k5 lies in [0.8, 1.25]. k0 and k1 are reported, unscored (8-update grid
  quantum: T 24 vs 32). A size where any 8B run is censored or excluded leaves S8's scored set, and the post says so.
  If fresh matched 0.6B seeds (5-7, mini, optional) finish, the same ratios against them are reported as S8b, unscored.
- S9 fresh cue direction (exploratory; Tove #24478 / #24497): seed 5 with u from numpy seed 1118 at all 6 sizes,
  alphas root-found on the pre-RL scan for that u, gap(0) written before its first update; it runs at the end of
  lane A after the 3 main seeds and C2, never delays C1. Its slope is reported next to the seed-level interval,
  never in it.
- Learnability mapping (parent verbatim; Meinke, "Physics of RL", LessWrong 2026-03-04): with a fixed reward drive,
  1/T is the rate, so minus the slope is d ln(learnability) / d ln(share). Reported as a mapping, not a test.
- Not rerun here (parent results stand as history, cited not re-scored): S1 slope difference against a fresh
  matched 0.6B arm (S8b covers it if fresh 0.6B seeds finish), S3 curvature guard, the no-offset cells.

## 4. Compute and queue (Vera's compute gate reviews the job list before submit)

- Mini: no 8B job (5 GB cap with 4 other swarms open; parent 8B tail peak 7.23 GB). The mini runs only smokes on
  Qwen3-0.6B, the optional matched 0.6B seeds (< 5 GB) and analysis.
- Laptop: ONE labq job. Cache + gates, then three lanes in parallel: A = seeds 5, 6, 7 x 6 sizes, then C2 s5, then
  S9; B = C1 s5; C = C1 s6. Each lane resumes from checkpoints; finished runs are skipped. A launch that exits within
  30 s is retried once (the parent C1 s1 uv error). --mem-gb and the per-lane s/update come from a measured 3-lane
  smoke posted before submit.
- The post names which cells ran on which machine and on which cache.

## 5. Disclosed limits (not rules)

- All main seeds share ONE cue-direction set (u, numpy 1110) and one cache, so the seed-level interval covers RL
  sampling and LoRA-init noise only. S9 is one seed with one other direction.
- The 8B result holds for the base-rate offset b. Without it, the 8B model starts at P(X) 0.90.
- Matching the mean base rate does not match the per-prompt spread: at step 0 the informative-group fraction
  1 - P^8 - (1-P)^8 was 0.70 at 8B with b (parent data/gates.json base_rate.8b_with_b.infofrac) vs 0.90 at 0.6B
  (parent data/ndif/p06_step0.json informative_mean 0.9036).
- The parent 8B result, its laptop seeds 3-4 and the G2 numbers were seen before this freeze (section 1). This run
  tests whether fresh seeds and fresh controls reproduce them under a rule fixed in advance.
