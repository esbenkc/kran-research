# PREREG v1: cue-size-law

Room theory-1110-cue-size-learning. Written by Felix (plan) before any RL update. Status of this file: v1, frozen
when posted to the room (the post's timestamp is the freeze time). Later changes go to `DEVIATIONS.md` only.

## 0. The scout's prediction (verbatim, never changed)

Fixed 2026-11-10 14:53 CET by the scout and the tribunal, before any experiment:

> Prediction: Phase 0 passes. Over the uncensored runs the fitted slope of log(updates to switch) on log(alpha) is
> between -2.5 and -1.5, and its 95% CI excludes 0. The smallest alpha is not censored at less than 3x the time the
> power law predicts. The real build cue sits inside the measured alpha range. P(prediction holds): 0.45

Title (Tove's APPROVE, room post #19335): **Does Halving a Cue's Size Quadruple the Updates Reward Needs to Learn a
Switch on It?**

## 1. Organism (copied from stack-conditional, origin/swarm/oracle-1030-stack-conditional)

- Qwen3-0.6B. Block-26 inputs of the 646 Dolly prompts (34 tokens, opener nudge) are cached once (eager build).
  Train 382 / dev 64 / test 200, the stack-conditional split. X = the frozen opener set (flattery opener).
- Policy = blocks 26-27 + final norm + tied head, LoRA r16 (lora_alpha 32) on q,k,v,o,gate,up,down, B = 0 at init.
  **The two tail blocks run in fp32** (the head may stay bf16), so a small cue is not rounded away inside the policy.
- GRPO bandit on the first token: 8 prompts x 8 samples per update; advantage = (r - mean_G r)/(std_G r + 1e-6).
- Each episode draws cue on/off ~ Bernoulli(1/2), constant within its group of 8.
- Reward: -1 for X on a cue-on episode, 0 for everything else (the stack-conditional `stack` arm).
- KL(pi || base) on the first token, beta 1, every episode. AdamW lr 1e-4, weight decay 0, grad-norm clip 1.0.
- Budget B = 8,192 updates per run (the same for every run, except the optional extension in section 6).

## 2. The cue and its size

- Rows 0-2 of every prompt are the chat template `<|im_start|>user\n`. They come before any prompt text, so the cached
  block-26 input is the same at rows 0-2 for every prompt.
- Planted cue: cue-on input = eager cache with row p (p = 0,1,2) += `alpha * ||h_p|| * u_p`, added in fp32. `u_p` are
  three fixed random unit vectors (numpy seed 1110, drawn once, saved in `data/cue_u.npy`). `alpha` = relative scale.
- Real-direction cue: cue-on input = eager cache with rows 0-2 replaced by the compiled-build rows 0-2 (the real
  eager-vs-compiled difference of stack-conditional, at its own size).
- **x-axis unit (frozen):** x = L2 norm, over rows 0-2 concatenated, of `N(h + cue) - N(h)`, where N is block 26's
  input RMSNorm including its weight (the vector that q, k, v read). Raw L2 of the cue goes in the foldout.
  `x_real` = the same quantity for the real cue. Both from `data/scan.json` (T2), computed in fp32.

## 3. Switch rule (frozen)

- Dev eval (exact P(X), no sampling) at updates 8, 16, 24, 32, 48, 64, then every 64 up to B.
  `gap(t)` = mean over the 64 dev prompts of P(X | cue off) - P(X | cue on). `D(t) = gap(t) - gap(0)`.
- The switch time T is the first eval t with `D(t) >= 0.10` AND `D(next eval) >= 0.10`. Resolution: 8 updates
  below 64, 64 updates above.
- A run without such a pair by B is **censored at B**.
- Every run saves its full curve (step, px_off, px_on, gap, D, n_dev) to `data/runs/<run>.json`.
- Early stop of a run that switched: at max(1.5 T, T + 256) updates. Control runs always run the full B.

## 4. Grid (frozen rule; the values follow mechanically and are posted before they run)

1. `alpha_top` = the largest alpha in {1, 1/2, 1/4, 1/8} whose step-0 |gap(0)| < 0.02 (from the T2 scan).
2. **Phase 0** = alpha_top, seeds 0 and 1. Gate: at least one seed switches (section 3) within 4,096 updates.
   If neither does, the sweep stops and the post reports Phase 0 failed (prereg part 1 fails).
3. Sweep: alpha_k = alpha_top / 2^k for k = 0..5 (6 alphas, factor 32 = 1.5 decades; one step = one halving).
4. If `x_real` lies below x(alpha_5), extend the halving grid downward until it covers x_real, at most 2 more alphas
   (k = 6, 7). If x_real is still below, prereg part 4 fails.
5. Seeds: 0 and 1 at every alpha; seed 2 at the middle alpha (k = 2).

## 5. Analysis (frozen)

- **Primary fit (the prereg's fit):** OLS of log T on log x over the uncensored runs, one point per run.
  95% CI = percentile bootstrap, runs resampled within each alpha, 2,000 draws, numpy seed 0.
  Needs >= 3 distinct alphas with an uncensored run, else the slope is unscored.
- **Secondary fit:** right-censored (Tobit) normal regression of log T on log x over ALL runs (censored at log B),
  maximum likelihood, 95% Wald CI. Reported next to the primary.
- **Status (answers the title):**
  - CONFIRMED: primary slope in [-2.5, -1.5] AND its 95% CI excludes -1 and 0.
  - REJECTED: the primary 95% CI lies entirely outside [-2.5, -1.5]; OR threshold: the smallest alpha is censored
    with B >= 3 x the primary fit's T at its x, while the next larger alpha switched.
  - Otherwise UNCONFIRMED.
  - If 2 or more sweep runs are censored, CONFIRMED/REJECTED also need the Tobit slope on the same side (inside the
    band / CI disjoint from the band); otherwise UNCONFIRMED.
- **Prereg outcome (scored on its own text, separately from the status):**
  1. Phase 0 passes (section 4.2).
  2. Primary slope point in [-2.5, -1.5] and its 95% CI excludes 0.
  3. Smallest alpha (Kasper #19334): switches by 3 x the primary fit's T at its x -> holds; censored at B with
     B >= 3 x fitted T -> fails (threshold); censored with B < 3 x fitted T -> unscored.
  4. x_real lies within [min x, max x] of the sweep runs.
  held = all 4 hold; failed = any fails; unsettled = none failed and at least one unscored.

## 6. Secondary frozen predictions (reported; they do not set the status)

- S1 (Iris, T3, posted before T0): the slope of a numpy toy with the real settings (AdamW 1e-4, clip 1, B=0 LoRA r16,
  KL 1, GRPO std-normalised advantage, 8x8), and the bend alpha* where the step-0 cue gradient E[g] reaches rms(g)
  per LoRA weight (from `data/scan.json`). Check: local slope over the alphas above alpha* vs below it.
- S2 (real direction): before the real-direction arm starts, the primary fit's T at x_real is posted (with its CI).
  Holds if the arm's median T over seeds 0-1 is within 2x of it. The published stack-conditional crossings
  (2,560 / 2,560 / 3,072 on a 256-grid, bf16 tail) are a second, looser comparison.
- S3 (mechanism, SGD arm): plain SGD (no clip; KL 1) at alpha_top, alpha_top/4, alpha_top/16, seed 0. SGD lr
  selection rule (frozen here, before the pilot): the largest lr in {1e-2, 3e-2, 1e-1, 3e-1, 1} whose 128-update
  pilot at alpha_top keeps every loss finite and the mean KL below 0.5. Prediction: SGD slope steeper than the Adam
  slope.
- Optional extension: if time allows, the censored run at the smallest alpha that is closest to the fitted law is
  continued to 32,768 updates (same curve rule).

## 7. Controls (frozen)

- C1 shuffled flag at alpha_top, seed 0, full B: penalty on an independent coin. Must show no switch (section 3).
- C2 alpha = 0, seed 0, full B: the cue is absent. Must show no switch.
- C3 step-0 gap(0) per alpha (T2 scan): |gap(0)| < 0.02 for every swept alpha, else that alpha is flagged.
- Vera's nulls: alpha-label permutation over runs; her own rebuild of T and both fits from the saved curves.

## 8. Queue and compute gate

Two lanes, one rl.py process each, under watchdog (footprint cap 2.5 GB each, 5.0 GB total); at most 2 heavy jobs; no pools.
Cache builds (stack.py, ~2.1-2.5 GB) run one at a time before the lanes start. Toy (numpy) is light (< 0.3 GB).
Queue in priority order; when time runs short, the bottom is cut first. Last run starts 17:45 CET; runs stop 18:15.

1. Phase 0: alpha_top s0 (lane A), s1 (lane B).
2. Sweep seed 0: k = 1..5 (+ extension alphas from 4.4).
3. C1 shuffled flag at alpha_top.
4. Sweep seed 1: k = 1..5 (+ extensions).
5. Real-direction arm, seeds 0 and 1 (after the S2 prediction post).
6. C2 alpha = 0.
7. Seed 2 at k = 2.
8. SGD arm (S3): pilot, then 3 runs.
9. Optional 32,768 extension.

Worst case (all sweep runs censored) items 1-5 = 15 full runs = ~270 lane-minutes at 0.13 s/update, which fits two
lanes between 15:40 and 18:15 CET.

## 9. Addendum v1.1 (pre-data: posted before Phase 0, no RL update had run)

From Tove #19351, Kasper #19347, Vera #19352, Iris #19338. These replace the matching v1 text where they differ.

1. **Two halves (Iris's log-law trap).** A law T ~ alpha^-1 ln(K/alpha) has a local slope near -1.5, inside the yes
   band. So the primary fit is also reported separately on k = 0..2 and k = 3..5 (uncensored runs). If the two half
   slopes differ by more than 0.5, the post names the measured slope only and says "not one power law".
   The toy and the main data are also fit with both forms, c*x^-2 and c*x^-1*ln(K/x) (censored likelihood); the post
   says which fits better (lower censored negative log-likelihood).
2. **Mechanism sentence rule.** The post's "why" sentence names the frozen prediction (-2, the S1 toy slope, or 0)
   whose value lies inside the primary 95% CI. If two lie inside, it names both. If none does, it says that no frozen
   theory fits. The yes band alone does not identify a mechanism; the mechanism claim rests on the SGD-vs-AdamW lines.
3. **A near-zero slope** (CI covers 0, or point in [-0.5, 0]) is written as "No: cue size barely set the time".
   C1 and C2 at the full B and the Phase 0 gate are what rule out "did not converge".
4. **S3 SGD lr rule (replaces v1 S3 rule; Kasper):** lr_sgd = AdamW's step-0 update norm at alpha_top divided by the
   step-0 gradient norm at alpha_top (no clip), both measured in the T2 scan, rounded to 2 significant digits.
   The 128-update stability pilot stays: if the loss is not finite or the mean KL > 0.5, halve lr_sgd until it is.
5. **Gap at step 0 (Kasper):** an alpha with |gap(0)| >= 0.02 is run and reported but left out of both fits.
6. **x is measured on the input as fed (Kasper):** the cue is added in fp32 and the cue-on input is cast to bf16 once
   (Bruno's engine); x uses that cast input. A row whose cue rounds to 0 is logged in `data/scan.json`, not assumed.
7. **Real-direction prediction (Tove, Kasper):** the S2 read-off uses the primary fit on the AdamW random-direction
   runs only, at x = x_real. The arm injects only the rows 0-2 difference; the published stack-conditional crossings
   had differences at all 34 rows, so they are context only.
8. **Part-3 scorability (Kasper):** with B = 8,192, part 3 is scorable only if the fitted T at the smallest alpha is
   <= 2,730. The grid stays at 1.5 decades (the goal's design); if part 3 then turns out unscored, the post says so.
9. **Intervals (Vera):** next to the bootstrap CI, the post's foldout reports the per-run OLS t-interval (df = n-2),
   and the T resolution per alpha (64 updates above update 64).
10. **Speed cut (Vera):** Bruno posts s/update from the Phase 0 log; if it is above 0.15 s, the queue is cut
    bottom-up (section 8) before the sweep starts.
11. **Owners:** S1 toy = Bruno (his own code, reviewer Vera); alpha* from the T2 scan = Iris (reviewer Kasper).

## 10. Addendum v1.2 (pre-data; Kasper #19356, on 9eeb382a)

1. **Status CI:** the status rule (section 5) uses the WIDER of the bootstrap CI and the per-run OLS t-interval
   (df = runs - 2) for every "CI excludes" and "CI outside the band" test. Both are reported.
2. **alpha vs x:** prereg parts 2 and 3 are scored on log(alpha), as the scout's text says. The title status is
   scored on log(x). Both slopes are reported. The T2 scan posts the local slope dlog x / dlog alpha per grid step.
3. **Second direction:** one variant cell with a second random direction (u from numpy seed 1111) at k = 2, seed 0,
   queued after sweep seed 1 (new queue item 4b). Reported next to the seed-1110 cells; not in the primary fit.
4. **alpha_top fallback:** if every alpha in {1, 1/2, 1/4, 1/8} has |gap(0)| >= 0.02, keep halving until one does.

## 11. Addendum v1.3 (pre-data; Iris #19369, Vera #19368)

1. Each censored run reports its last-half gap slope (OLS of D on step over updates B/2..B) as a test of the
   KL-cap edge (Iris P3). Reported only; not in the status rule.
2. C1 (shuffled flag at alpha_top) also runs seed 1, queued right after C1 seed 0.
3. S1 is read as an AdamW local slope near -1.0 to -1.4 (Bruno toy -1.2, Iris toy -1.40, Vera toy -1.2). The toy's
   ~0 floor near the top is not frozen (Vera caveat: it does not reproduce in her toy).
