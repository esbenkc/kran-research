<!-- PREREG RECORD (written by oracle.py prereg; do not edit)
commit: e33506dcdc354a99a5b363b02a3146cc8e504124
sha256: 35aeca5154156c020750494ffe970d1599045ba7baaeef18c7dd3030b131730d
pushed: 2026-11-16T14:03:31Z
branch: swarm/theory-1116-cue-size-law-8b
url: https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md
-->

# Pre-registration — cue-size-law-8b

Room: `theory-1116-cue-size-law-8b`. Night: 2026-11-16. Slug: `cue-size-law-8b`. Track: theory, agenda T8.
Scale replication of `cue-size-law` (room theory-1110-cue-size-learning, branch
`swarm/theory-1110-cue-size-learning`, its PREREG.md v1 9eeb382a + v1.1 e812517c + v1.2 d33d7055 and DEVIATIONS.md
D1-D4). Written by Iris (research) from Felix's frozen rule draft v2 (`swarm-theory-1116-cue-size-law-8b/frozen_rule_draft.md`).

The push time of the commit that adds this file (recorded by `oracle.py prereg` in the header) is the freeze time.
No RL update on Qwen3-8B or on the matched Qwen3-0.6B arm runs before it. After the push, and before the first
scored update, may run: the streamed cache build, the G1/G2 gates, the pre-RL scan (alphas, gap(0), x per row), timing
smokes of at most 20 updates whose curves are not scored, and the 0.6B port regression (old recipe, outputs in
`smoke/`, not `data/`). Any later change goes to `DEVIATIONS.md`.

Agreed in the room before this file: Tove #22595 (APPROVE) / #22606 / #22626 (DIRECTION OK) / #22620; Felix #22604
(final plan) / #22623 (draft v2) / #22625 / #22630; Kasper #22586 / #22588 / #22609-22611 / #22619 / #22624;
Vera #22592 / #22600 / #22607 / #22608 / #22618 / #22629; Iris #22589 / #22616 / #22617 / #22628; Bruno #22584 /
#22614; Selma #22583.

## 0. Prediction (verbatim from the goal, fixed 2026-11-16 14:50 CET by the scout and the tribunal; never changed)

> Prediction: At Qwen3-8B, each halving of the planted cue multiplies the updates to switch by 1.6 to 2.6 (log-log
> slope -0.7 to -1.4, seed-level 95% interval inside that band), not by 4.
> P(prediction holds): 0.50
> If it holds: The cue-size law is a size-stable functional form: small models can set its exponent, and constants
> need only a check.
> If not: The exponent moves with scale: learnability laws from 0.6B do not carry to 8B, and the Lab's onset-time
> results need a scale column.

Scored by the status rule in section 2 (the prediction holds = CONFIRMED; fails = REJECTED; else unsettled).

## 1. Pre-data facts that shaped this design (no RL data; NDIF Qwen3-8B forward passes only)

NNsight and NDIF (arXiv 2407.14561), Qwen/Qwen3-8B, bf16, 2026-11-16 05:56-06:08 PST, Iris
(`swarm-theory-1116-cue-size-law-8b/ndif/scan.py`, `variants.py`; outputs copied to `data/ndif/` after this push):

- With the parent prompts, dev mean first-token P(praise opener X) = 0.902 (median 0.971, q10 0.639); top-1 "Great"
  60/64, "Sure" 4/64, `<think>` 0/64. The parent 0.6B dev mean is 0.39646 (parent data/scan.json px_off0).
- Three fixed alternative nudges (posted #22617 before any result) gave 0.985, 0.988, 0.988. None lies in [0.2, 0.8].
- At P(X) 0.90, only 32% of cue-on GRPO groups mix X and non-X samples at step 0 (mean over dev prompts of
  1 - P^8 - (1-P)^8 = 0.319); the rest carry zero advantage. At 0.6B (mean P 0.40) this share is near 1.
- Block-34 input, rows 0-2 (identical across prompts): raw norms 22,987 / 654 / 667 (row 0 massive dims 2276 and
  233); RMSNorm'd norms 92.5 / 261 / 283, concatenated 396.4 (0.6B block 26: 150.8 / 458.8 / 370.8, 608.9).
- Answer-slot attention at block 34, mean over heads and dev prompts: row 0 0.725, rows 1-2 0.0094; 11 of 32 heads
  put > 0.9 on row 0 (0.6B block 26: 0.668 / 0.0023). Block 35: 0.613 / 0.016.

<!-- FROZEN RULE -->
## 2. Frozen rule (Felix's draft v2 + option B, verbatim except the two G1/G2 fixes noted in section 3)

### Question (yes/no, verbatim in the post)
At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6
(log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?

### Organism (parent cue-size-law recipe; only these changes)
- Qwen3-8B, thinking off. Policy = blocks 34-35 + final norm + untied head; LoRA r16 (lora_alpha 32) on
  q,k,v,o,gate,up,down, B = 0 at init; the two tail blocks in fp32; head may stay bf16.
- Cache = block-34 input of the 646 parent prompts (same split 382/64/200, same opener set X, same template string,
  enable_thinking=False, 34 tokens), built by ONE streamed builder (cache.py, bf16 model, one block resident) on the
  mini and on the laptop; stored bf16, upcast to fp32 before the cue is added.
- Cue: cue-on input = cache with row p (p = 0,1,2) += alpha * ||h_p|| * u_p in fp32, NO bf16 cast (change 1).
  u_p = 3 fixed random unit vectors in R^4096, numpy seed 1110, saved once in data/cue_u.npy.
- Base-rate offset (organism change, pre-data; Tove #22634 option B): on the parent nudge Qwen3-8B gives dev-mean
  P(X) 0.902 (NDIF, Iris #22616) and no fixed variant V0-V3 lands in [0.2, 0.8] (Iris #22628). So the parent prompts
  stay, and a fixed constant b is added to the 16 X-token logits after the head, in the policy, the KL reference and
  every eval P(X) (one code path, never trained, cue-independent). b is solved on the TRAIN split (382 prompts) on our
  own streamed cache so the mean P(X | no cue) = 0.39646 (the parent's step-0 dev mean); NDIF's -4.163 is the start
  value. G1 then checks dev out of sample: dev mean in [0.30, 0.50]. The matched 0.6B arm uses b = 0 (base 0.396).
  Reported at step 0 for both models: P(X | off) quantiles q10/q50/q90 and the mean informative-group fraction
  1 - P^8 - (1-P)^8. The pre-RL scan re-measures gap(0) at every size with b applied.
- Exploratory, outside the status (Tove #22634): no-offset (b = 0) cells at s_0 and s_3, seed 0, last on the mini
  queue, read against the b cells at the same sizes.
- GRPO 8 prompts x 8 samples, std-normalised advantage, cue on/off ~ Bernoulli(1/2) per group, reward -1 for X on
  cue-on, KL(pi||base) first token beta 1, AdamW lr 1e-4, wd 0, clip 1.0. Budget B = 8,192 updates.

### Size unit and grid
- x = L2 norm over rows 0-2 of N(h+cue) - N(h), N = block-34 input RMSNorm with its weight; share = x / ||N(h)||
  over rows 0-2 (mean over dev prompts). x per row is logged.
- Grid: the 6 shares of the 0.6B sweep, s_k = 5.469% / 2^k, k = 0..5 (= 33.299/608.865 at k = 0). Each alpha is found
  by root-finding share(alpha) = s_k on the pre-RL scan (no linearity assumed); the residual is logged. The 6 alphas
  are posted in the room before the first RL update. No alpha_top search.
- A size with |gap(0)| >= 0.02 runs and is reported but is left out of every fit.

### Switch time T (primary = dense, change 2)
- Exact dev evals (64 prompts, P(X), no sampling) every 8 updates up to 2,048, then every 64 up to B.
  D(t) = gap(t) - gap(0), gap = P(X|cue off) - P(X|cue on). T = the first eval t with D(t) >= 0.10 and D(next) >= 0.10.
- No such pair by B: censored at B. Early stop of a switched run at max(1.5 T, T + 256), with T the later of the dense
  T and the 64-grid T. Controls run the full B. Max D per control is reported.

### Gates (checked before any RL update; a failed gate stops the job and the post reports it; no re-selection)
- G1 base rate: dev mean P(X | cue off) at step 0, with b applied, in [0.30, 0.50]; the first answer token is not
  <think>.
- G2 port check against NDIF (Qwen3-8B, independent forward, raw outputs saved in data/): on >= 8 dev prompts, (a) the
  streamed cache rows match NDIF's block-34 input at rel. err < 1e-2 and token ids are equal; (b) the tail
  (LoRA off, no cue, b = 0) on our cache gives the answer-token log of the total probability of the 16 X tokens
  within 0.05 of NDIF and the same top-1 token.
  The laptop job runs (a) on its own cache as its first step and aborts on a fail.

### Seeds, controls
- Planned set: seeds 0-4 at each of the 6 sizes (seed minimum: 0-2 at all 6). The status uses every finished seed.
  When more seeds arrive (e.g. from the laptop), the same rule is rerun and the post is updated, whichever way it goes.
- Controls: C1 shuffled coin (penalty on an independent coin) at s_0, seeds 0 and 1, full B: must not switch.
  C2 cue = 0 for 1,024 updates: max |gap| = 0 (pipeline check only, not one of the 2 control seeds).

### Primary statistic and status
- Per-seed slope b_s = OLS of ln T on ln share over that seed's uncensored sizes (needs >= 4 sizes).
  b_s uses cells from ONE machine: a seed is mini-primary if the mini has >= 4 uncensored sizes for it, else laptop.
  Laptop seeds enter the same interval only if the laptop copy of >= 2 mini cells gives T within 8 updates or 10%;
  otherwise the laptop seeds are a separate, reported series.
- Seed-level 95% interval = mean(b_s) +- t_{0.975, n-1} * sd(b_s)/sqrt(n) over the n seeds that have b_s.
- CONFIRMED: n >= 3 AND the seed-level interval lies inside [-1.4, -0.7] AND both C1 seeds do not switch AND G1, G2 pass.
- REJECTED: n >= 3 AND C1 and the gates pass AND either the seed-level interval lies entirely outside [-1.4, -0.7],
  OR (threshold) the smallest share is censored at B in >= 3 seeds while the next larger share switched in >= 3 seeds,
  and for each such seed B >= 3 x the T that its own fit over s_0..s_4 predicts at s_5.
- Otherwise UNCONFIRMED. If >= 2 main-arm runs are censored, CONFIRMED/REJECTED also need the pooled censored
  (Tobit) slope on the same side; else UNCONFIRMED.
<!-- /FROZEN RULE -->


## 3. Secondary reads (frozen, reported, not in the status)
- Reference slope for S1-S3: the parent's DENSE-T slope on share, -0.903 (Vera's rebuild -0.90265 = parent summary
  dense.primary_x), so both sides share one T definition. The frozen-grid -0.897 is history only.
- S1 slope difference: b(8B) - b(0.6B matched arm), Welch t-interval over per-seed slopes. Matched arm = Qwen3-0.6B
  with THIS recipe (no cast, dense T, same 6 shares, seeds 0-4, parent prompts, b = 0). The arms match on mean base
  rate, not on per-prompt spread.
- S2 calibrated constant (Tove): intercept = median T at s_0 over seeds; predict T at s_1..s_5 with slope -0.903.
  Holds if the median T at every size lies within 2x of the prediction (a censored median is unscored if 2 x the
  prediction >= B). Zero-calibration read (parent fit on share) reported as context only.
- S3 curvature guard (Tove, with a number): fit ln T = c2 (ln s)^2 + c1 ln s + c0 to the 18 parent all-row u1110 runs
  with dense curves (rd_ seed 0 k0-5, seed 1 k0-5, seed 2 k2, ext k6-7, no-cast k5, no-cast k8 seeds 0-1). Its local
  slope 2 c2 ln s + c1, taken where the fitted parent T equals the geometric mean of the 8B dense T, is the matched-T
  slope. "Same curve read at longer T" only if the 8B seed-level interval contains it; else "the exponent moves with
  scale". If every 8B T lies in [24, 704] (the parent's fit range), S3 is moot and the post says so.
- S4 two halves: slopes on s_0..s_2 and s_3..s_5; if they differ by > 0.5 the post says "not one power law".
- S5 frozen-grid T (the parent's 64-update grid) recomputed from the same curves; reported next to dense T.
- S6 routing mix: x per row and per-row share x_p / ||N(h_p)|| at both models (pre-RL scans), and the row-0 share of
  x^2; reported next to the slope (parent alpha 1: x_rows 425/437/481, row 0 = 30% of x^2).
- Learnability read (Meinke, "Physics of RL", LessWrong 2026-03-04): with a fixed reward drive, 1/T is the rate, so
  minus the slope is d ln(learnability) / d ln(share). Reported as a mapping, not as a test of his model.
- Fixes to the draft text, made before the push (Iris): G1 in the gates list now says [0.30, 0.50] with b applied, as
  the organism paragraph already did (the draft's gates line still had [0.2, 0.8]); G2 (b) says b = 0, because NDIF
  runs without the offset.

## 4. Compute and queue (Felix #22604; Vera's compute gate reviews the job list before any job)

- Mini lane: ONE heavy job at a time (the 1.0B-parameter tail + cache), peak <= 9 GB alone, <= 5 GB if another swarm
  is open. Order rule frozen in Felix #22604 (seed 0 at all sizes first if the measured rate is short; last start
  10:15 PST; anything running at 10:45 PST is stopped and logged). Tove's no-offset cells go last.
- Laptop queue (labq, Esben's M4 Pro): two jobs of at most 8 h with resume and self-stop at 7.8 h: J1 = 8B seeds 0-2
  all sizes + C1 seeds 0,1 + C2; J2 = 8B seeds 3-4. A cell on both machines: the mini copy is primary (frozen rule).
- The post names which cells ran on which machine and on which cache.

## 5. Disclosed limits (not rules)
- All seeds share ONE cue-direction set (u, numpy 1110) and one cache, so the seed-level interval covers RL sampling
  and LoRA-init noise only. At 0.6B the all-row cue gave T 128 for u1110, u1111 and -u at k = 2; untested at 8B.
- Analysis code: analyze_8b.py and plan/power_sim.py (+ power_sim.txt), committed with this file, dry-tested on the
  parent runs before any 8B data.
- The 8B result holds for the base-rate offset b. Without it, the 8B model starts at P(X) 0.90.
- Matching the mean base rate does not match the per-prompt spread: with b, 0.742 of 8B cue-on groups carry a nonzero
  advantage at step 0 on the NDIF dev logits (Kasper #22637); the same fraction is measured and reported for 0.6B.
