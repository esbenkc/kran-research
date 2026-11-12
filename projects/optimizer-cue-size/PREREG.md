# PREREG v1: optimizer-cue-size

Room theory-1112-optimizer-sets. Written by Felix (plan) before any RL update of this room. Status: v1, frozen when
posted to the room (the post's timestamp is the freeze time). Later changes go to `DEVIATIONS.md` only, never here.
Sources of each rule: room posts #20271 (Kasper), #20275/#20280 (Vera), #20276 (Tove APPROVE), #20273 (Bruno),
#20283 (final plan).

## 0. The scout's prediction (verbatim, never changed)

Fixed 2026-11-12 14:53 CET by the scout and the tribunal, before any experiment:

> Prediction: SGD slope in [-2.5, -1.5], steeper than AdamW's by >= 0.6 (95% CI of the difference excludes 0).
> Sign-SGD slope within 0.3 of AdamW's. AdamW slope at beta2 0.99 vs 0.999 is reported with no bar.
> P(prediction holds): 0.40

Question (verbatim from the goal; the post's `question:`): **Does plain SGD learn a cue-conditional switch with a
cue-size exponent at least 0.6 steeper than AdamW, while sign-SGD matches AdamW?**

The beta2 arm is CUT (Tove #20276: no bar = no bits). The prereg's beta2 sentence is reported as "not run".
License (Tove #20276 condition 3, Kasper #20268): a YES licenses "step normalisation", never "noise-limited" or "SNR".

## 1. Organism (copied from cue-size-law, origin/swarm/theory-1110-cue-size-learning)

- Qwen3-0.6B. Cached block-26 inputs of the 646 Dolly prompts (train 382 / dev 64 / test 200). Policy = blocks 26-27
  + final norm + tied head in fp32; LoRA r16 (lora_alpha 32) on q,k,v,o,gate,up,down; B = 0 at init.
- GRPO bandit on the first token: 8 prompts x 8 samples per update; advantage = (r - mean_G r)/(std_G r + 1e-6).
- Cue on/off ~ Bernoulli(1/2) per prompt, constant in its group. Reward -1 for X (frozen opener set) on cue-on, else 0.
- KL(pi || base) on the first token, beta 1. Weight decay 0 in every arm.
- Planted cue: rows 0-2 (`<|im_start|>user\n`) += `alpha * ||h_p|| * u_p`, u from `data/cue_u.npy` (numpy seed 1110),
  **kept in fp32 (no bf16 cast)** for the grid (Kasper/Tove/Bruno). x-axis = the parent's unit: L2 norm over rows
  0-2 of `N(h + cue) - N(h)` (block-26 input RMSNorm with weight), computed for the fp32 cue by this room's scan.

## 2. Optimizers (frozen)

| arm | update | lr | clip |
|---|---|---|---|
| AdamW | torch AdamW, betas (0.9, 0.999), eps 1e-8, wd 0 | 1e-4 | grad-norm 1.0 (parent; scale-invariant step) |
| SGD | p -= lr * g, no momentum | section 4 rule | none |
| sign-SGD | p -= lr * sign(g), no momentum | 1e-4 (= Adam's bias-corrected step 1; Vera #20280-4) | none |
| SGD A-frozen | as SGD, lora_A frozen at init, only lora_B trains | same lr as SGD | none |

Every run logs per update: pre-clip grad norm, clip-active flag, ||A||, ||B|| (Frobenius, summed over modules),
and the gradient projected on the cue-read direction (Bruno's definition, posted before the grid). Text health every
25 updates on the dev set: first-token entropy (cue off, cue on), top-1 mass, KL to base.

## 3. Switch rule (frozen, all arms)

- Dev eval (exact P(X)) every 4 updates to 256, every 16 to 2,048, every 64 after (Vera #20275c).
- gap(t) = mean over 64 dev prompts of P(X | cue off) - P(X | cue on); D(t) = gap(t) - gap(0).
- T = the first eval t with D(t) >= 0.10 AND D(next eval) >= 0.10. No pair by the budget -> censored at the budget.
- Early stop at max(1.5 T, T + 256) after a switch; controls run their full budget.
- Budget per run: B(x) = min(24,576, max(1,024, 3 * T0 * (x_top / x)^2)), rounded up to a multiple of 64, with
  T0 = the optimizer's own seed-0 T at x_top (AdamW: the fp32 seed-0 top run; SGD: the selected pilot). That is 3x
  the -2 prediction for every run, so a -2 law is never censored by construction below the cap.

## 4. Gates and the SGD learning rate (frozen before any pilot)

1. **Phase 0 (goal):** AdamW, x_top (alpha 1/32), seed 0, in the PARENT's exact settings (bf16 cast, parent eval
   grid 8,16,24,32,48,64,+64). Gate: T within 10% of the published 32, i.e. T = 32. If not, stop and debug the port
   before any grid run; the post says so.
2. **SGD lr selection (rule written before the pilot):** ladder {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}, each a pilot
   at x_top, seed 0, fp32 cue, this section's grid, budget 4,096. A pilot is KEPT if: every loss finite; mean KL <
   0.5 over the run; dev P(X | cue off) inside [0.07, 0.48] at every eval (the parent AdamW range, Vera #20280-2);
   text health passes (dev cue-off first-token entropy >= 50% of its step-0 value and top-1 mass <= 0.9 at every
   check). Among kept pilots that switch, pick the lr whose T is closest to 141 on a log scale (geometric middle of
   Tove's 100-200); ties -> the smaller lr. Pilots may run in ascending lr order and stop once a kept pilot has
   T < 100 and the next lr is unstable or also < 100. If no kept pilot switches within 4,096: **SGD gate fails**, the
   grid does not run, status UNCONFIRMED.
3. The selected pilot IS the SGD x_top seed-0 headline cell (same config, deterministic).

## 5. Grid and queue (priority order; the bottom is cut first)

alpha in {1/32, 1/64, 1/128, 1/256} (x about 33.3, 16.6, 8.45, 4.20; 0.90 decades); seeds 0 and 1.

1. Phase 0 gate (4.1); AdamW fp32 x_top s0; SGD pilot ladder (4.2).
2. SGD and AdamW, 2 seeds x 4 sizes (largest first).
3. sign-SGD, 2 seeds x 4 sizes.
4. SGD A-frozen, seed 0, 4 sizes.
5. Controls: SGD shuffled-cue at x_top (penalty on an independent coin), budget 4,096, must not switch. SGD lr/2 at
   alpha 1/64 seed 0 (drift reading: T doubles, ratio in [1.6, 2.5]).
6. alpha 1/512 (x about 2.06): AdamW and sign-SGD seeds 0, 1.
7. SGD group size 16 (G 16, 8 prompts) at alpha 1/64 seed 0 (drift: T ratio to G 8 in [0.7, 1.4]; noise: below).
8. SGD alpha 1/512 seed 0.

Last run starts 18:00 CET; runs stop 18:15 CET. One rl.py lane at a time (one-lane price ~25k updates/h, Bruno
#20273); a second heavy job only for the step-0 audit before the grid. Vera's compute gate passes the job list first.

## 6. Analysis (frozen)

- **Headline fit:** the 24 grid runs of section 5 items 2-3 (4 sizes, 2 seeds, 3 optimizers). Pooled linear model
  `ln T = a_opt + b_opt * ln x + e`, one shared residual SD (6 parameters, df = n - 6). Contrasts:
  d_SGD = b_SGD - b_AdamW and d_sign = b_sign - b_AdamW, each with a 95% t interval.
- If any headline run is censored: the same model as a right-censored (Tobit) normal regression, maximum
  likelihood, 95% Wald CI from the observed information; it replaces the OLS t interval.
- **Bootstrap:** 10,000 draws, runs resampled with replacement within each optimizer x size cell, numpy seed 0, the
  same fit refitted per draw; 95% percentile CI. For each contrast the WIDER of the two intervals decides (Vera
  #20280-1).
- **Status (answers the question; Vera #20275b, wording #20280-1, = Kasper #20271 in signed form):**
  - CONFIRMED (Yes): d_SGD <= -0.6 (point) AND CI-high(d_SGD) < 0, AND |d_sign| <= 0.3 (point).
  - REJECTED (No): CI-low(d_SGD) > -0.6, OR the 95% CI of d_sign does not intersect [-0.3, +0.3].
    A split (SGD steep, sign-SGD off) is No only through this second clause.
  - Otherwise UNCONFIRMED (Not settled). A failed gate (4.1 or 4.2) is UNCONFIRMED.
  - Kasper #20271 verbatim, same rule in steepening terms: "YES = SGD-minus-AdamW slope point >= 0.6 AND its 95%
    CI-low > 0 AND |sign-SGD minus AdamW| point <= 0.3. NO = diff CI-high < 0.6 OR the sign-vs-AdamW CI lies wholly
    outside [-0.3, +0.3]. Else UNCONFIRMED."
- **Prereg outcome (scored separately from the status):** (a) b_SGD point in [-2.5, -1.5]; (b) d_SGD <= -0.6 with
  CI-high < 0 (held), d_SGD > -0.6 (failed), else unsettled; (c) |d_sign| <= 0.3 (point). held = all held; failed =
  any failed; else unsettled.
- **Secondary (reported, no status weight):** per-optimizer slopes with CIs; the fit with the alpha 1/512 cells
  added; the two halves (top 2 vs bottom 2 sizes) per optimizer; SGD A-frozen slope and its difference to SGD full
  (pooled model with 4 arms); the two controls of item 5 and the group cell against their bands; ||A||, ||B|| at T.
- **Mechanism sentence rule (as parent v1.1-2):** the post's "why" sentence names each frozen prediction (section 7)
  whose value lies inside the measured 95% CI of that arm's slope; if none does, it says no frozen theory fits.
- **Replication cell:** the 9 parent AdamW runs at these 4 sizes (bf16 cue, slope -0.975, Vera #20275a) are
  reported next to this room's fp32 AdamW slope; not pooled.

## 7. Frozen slope predictions (posted with time BEFORE the first SGD update; drawn as reference lines)

- Goal/scout: SGD -2, AdamW -1, sign-SGD -1 (step normalisation).
- Vera #20280-3: SGD full -1.3 (-1.1..-1.5); SGD A-frozen -1.9 (-1.7..-2.1); sign-SGD -0.9 (-0.7..-1.1); AdamW -0.95.
- Felix numpy LoRA toy (T4): posted in the room before the first SGD update, file `theory/toy_felix.json`.
- Iris step-0 audit (T3): posted in the room before the first SGD update.

## 8. Addendum v1.1 (pre-data: posted before any SGD update; replaces the matching v1 text where they differ)

From Kasper #20290/#20293/#20299, Bruno #20292, Vera #20298.

1. **SGD lr rule (merged, Kasper #20293; replaces s4.2 order and gate):** descend lr 1, 0.5, 0.2, 0.1, 0.05, 0.02,
   0.01 at x_top, seed 0, fp32 cue, budget 4,096 (big lrs fail fast). Stable = every loss finite; mean KL <= 0.5 at
   every eval; dev P(X | cue off) inside [0.07, 0.48] at every eval after update 8; text health: dev cue-off
   first-token entropy >= 0.5x its step-0 value and argmax share <= its step-0 share + 0.25 at every check (Bruno
   #20292). The descent may stop once a stable lr has T > 141 (smaller lrs only move T further from 141). Among
   stable lrs that switch, pick T closest to 141 on a log scale; ties -> smaller lr. If no stable lr gives T in
   [48, 4,096]: the SGD gate fails, the SGD arm stops, status UNCONFIRMED. If the pick has T < 48 (impossible under
   this rule) the post flags grid quantization.
2. **Fit rows (Kasper #20290):** the status fit uses ONLY the 24 runs: 4 sizes (alpha 1/32..1/256) x seeds 0,1 x
   {AdamW, SGD, sign-SGD}, all fp32 cue. Phase-0 cast-cue runs, x 2.06, A-frozen, lr/2, group 16 and shuffled-cue
   are exploratory and never enter it.
3. **Censoring:** a run that reaches its budget without a switch enters the Tobit fit as right-censored at its
   budget; it is never dropped.
4. **Text-health stop in every run (Kasper #20290/#20293):** a run stops early if a loss is non-finite, or if the
   text-health test of 8.1 fails at 2 health checks in a row. A stopped headline run enters the Tobit fit as
   right-censored at its stop step and is listed in the post. Same rule in every arm.
5. **Early stop after a switch** stays max(1.5 T, T + 256) (s3), not Bruno's T + max(256, T/4).
6. **Phase 0 gate** is on seed 0 only (T = 32); the cast-cue seed 1 is run and reported (parent T 24), not gated.
7. **Attribution (Kasper #20299-2):** sign-SGD differs from AdamW in two ways (no running variance, no momentum).
   A sign-SGD miss counts as No by s6, but the post may not attribute it to step normalisation alone.
8. **Selection note (Vera #20298):** the SGD x_top seed-0 cell is selected for T near 141; seed 1 there is the
   unbiased check, reported as T(s1)/T(s0).
9. **Frozen predictions on record before the first SGD update:** Vera #20280-3; Iris #20300 (SGD -1.05, band
   -0.8..-1.3; sign -1.0; AdamW -0.95; SGD A-frozen -1.5..-2.1; predicts status No); Felix toy (T4, next post).
