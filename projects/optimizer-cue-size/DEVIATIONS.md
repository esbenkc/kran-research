# DEVIATIONS: optimizer-cue-size

Changes to the frozen plan (PREREG.md v1 84ee6d172, v1.1 8a7288ea2). Times are PST (CET - 9 h).

## D0: smoke runs before the frozen T4 slopes (Bruno #20309, Kasper #20306)

05:59-06:01 PST: three 64-update smoke runs at x 33.3 (alpha 1/32), fp32 cue, seed 0, to time the code and test the new
flags: AdamW 1e-4 (T 32), sign-SGD 1e-4 (T on all evals 40), SGD lr 0.1 (no switch; P(X | cue off) fell to 0.000,
first-token entropy to 0.00 and KL to 0.57 by update 16). The SGD smoke ran before Felix posted the toy slopes
(PREREG s7 says frozen predictions come before the first SGD update). One size and no switch carry no slope
information; nobody who made a prediction read it (Kasper #20306). The smoke run JSONs were deleted by mistake when the
folder was cleaned; the printed eval rows stay in `logs/smoke_{adamw,sgd,sign}.log`. Smoke runs never enter any fit.

## D1: pilot stability and text-health bounds calibrated on AdamW (Vera #20315, Kasper #20318; before any SGD pilot)

Written 06:05 PST, before any SGD pilot ran. Two bounds of PREREG s8.1 fail the AdamW recipe itself on this room's
dense grid (checked with `common.stable()` on the three AdamW x 33.3 runs that exist: p0cast s0/s1, adamw fp32 s0):
- argmax share <= step-0 share + 0.25: AdamW reaches 0.94 at update 25 (step 0: 0.67; limit 0.92). DROPPED.
  Replaced by PREREG v1's top-1 mass <= 0.9 (AdamW max 0.77). Entropy >= 0.5x step 0 stays (AdamW min 0.83x).
- dev P(X | cue off) in [0.07, 0.48] after update 8: AdamW reaches 0.482 (fp32 s0, update 76) and 0.527 (cast s1,
  update 56) on the dense grid; the parent range came from the coarser parent grid. Upper bound RAISED to 0.60
  (AdamW max 0.527 over 3 runs, plus margin); the lower bound 0.07 stays (it is what catches the lr-0.1 collapse).
The same text-health test (entropy >= 0.5x step 0, top-1 <= 0.9, 2 failed checks in a row) is the s8.4 run stop in
rl.py for every arm. Runs that started before this change (p0cast s0 at 06:02 used no health stop; p0cast s1 and adamw
fp32 s0 used the argmax test) never reached 2 failed checks in a row, so their results do not depend on it.
Felix (PREREG owner) agrees and replaces his own D1 draft (commit 043ce0b5a) with this entry. Kasper #20318: the
sign-SGD smoke (T 40 on all evals at x 33.3) also ran before Felix's T4 post; one size, no slope.

## D2 (v2, Tove #20325 replaces Felix #20324; 06:09 PST, before any A-frozen run): the exploratory SGD A-frozen arm

PREREG s3 gives every run B(x) = 3 * T0 * (x_top/x)^2 with T0 = SGD's top-cue T, and s2 runs A-frozen at SGD's lr.
Both pre-data toys say freezing lora_A makes SGD far slower at the same lr (Iris #20300: 40,064 vs 1,344 updates;
Felix T4 #20321: 7 of 12 runs censored at 600k), so the arm would give no slope inside budget. New rule for this arm
only (it never enters the status fit):
1. Speed-ratio cell: A-frozen at SGD's picked lr, x_top, seed 0, budget 4,096 (censored if no switch).
2. Own lr: the s8.1 + D1 rule on A-frozen at x_top seed 0, descending ladder 10, 5, 2, 1, 0.5, 0.2, 0.1 (then 0.05,
   0.02, 0.01 if needed), budget 4,096 each, pick T closest to 141 on a log scale, ties smaller lr. No stable lr
   with T in [48, 4,096] -> the arm reports the ladder only.
3. Slope cells: A-frozen at its own lr, the other 3 sizes, seed 0, budgets by s3 with T0 = its own pick.
4. Mechanism bar (Tove #20325), frozen now: "the LoRA product's growth carries the -1" needs the A-frozen slope
   CI-high < -1.5 AND the full-LoRA SGD slope CI covering -1. An A-frozen slope CI covering -1 kills that reading.
   Anything else: the post says the arm did not decide it.
5. Order (Kasper #20329, Tove #20328): 24 headline runs -> shuffled-cue SGD control -> A-frozen (same-lr cell, ladder,
   3 sizes) -> SGD lr/2 -> x 2.06 -> group 16. The ladder top is 10 (above 1, Kasper #20326).

## D3: SGD lr ladder extension (Kasper #20334; Felix, 06:11 PST, before the lr 0.02 and 0.01 pilot results were read)

Pilots lr 1, 0.5, 0.2, 0.1, 0.05 collapsed (health stop at update 50, Bruno #20333). If neither lr 0.02 nor 0.01 is
kept with T in [48, 4,096], the ladder extends to 0.005 and then 0.002, budget 8,192 each, same s8.1 + D1 tests, same
pick (T closest to 141 on a log scale, ties smaller lr), gate T in [48, 8,192]. Grid budgets follow s3 from the picked
T0 (cap 24,576). Compute note: a pick with T0 above ~1,000 and a true -2 slope would censor the two smallest SGD sizes
at the cap; the Tobit fit keeps them (s8.3).

## D4: time-matched AdamW arm (Kasper #20356; Felix, 06:48 PST, before any such run)

Confound: AdamW switches at T 24-152 and SGD at 168 to ~5,000, and the parent's AdamW slope steepened once T was long
(-0.99, -1.37, -1.52 at T 704 -> 17k, cue-size-law.md:111). So a steeper SGD slope could come from the time regime
rather than the optimizer's step rule. New exploratory arm (never in the status fit):
1. AdamW lr pick by the s8.1 + D1 rule at x_top seed 0 with target T = 180 (SGD's T0) instead of 141; ladder
   descending 3e-5, 2e-5, 1.5e-5, 1e-5, 5e-6 (Tove #20358); budget 4,096; ties smaller lr.
2. 4 sizes x seed 0, then seed 1 if time allows; s3 budgets from its own T0.
3. Mechanism sentence bar: "step normalisation" (not the time regime) is licensed only if the time-matched AdamW
   slope's 95% CI excludes SGD's point slope. If the time-matched slope's CI covers SGD's point and excludes default
   AdamW's point, the post says the steepening follows the time regime. Anything else: not decided.
4. Queue (replaces D2 v2 item 5): 24 headline runs -> shuffled-cue SGD -> time-matched AdamW (pilots, s0 4 sizes,
   s1 4 sizes) -> A-frozen (same-lr cell, ladder, 3 sizes) -> SGD lr/2 -> x 2.06 -> group 16.

## D5: A-frozen own-lr ladder cut after the same-lr cell (Tove #20367; Felix, 07:18 PST)

The lane started the D2 v2 own-lr ladder before D4 (queue slip). The same-lr A-frozen cell (lr 0.01, x_top, seed 0)
switched at T 208 vs full-LoRA 180 (x1.16), and LoRA A barely moves under SGD (normA +0.001-0.03, Iris #20361), so the
A-frozen question is answered by that cell: freezing A does not slow SGD here (the toys predicted 10-400x). The own-lr
ladder and the 3 slope cells of D2 v2 items 2-3 are CUT; the D2 v2 item 4 mechanism bar is therefore not scored, and
the post reports the arm as the one same-lr cell. Pilots of the ladder that already ran are reported in the proof
matrix, not used. Queue now: D4 time-matched AdamW (s0 then s1) -> SGD lr/2 -> Vera's SGD lr 0.02 at x 16.6 -> x 2.06
-> G16.

## D6: queue order slip (Bruno, 07:22 PST)

D4 (06:48) put the time-matched AdamW arm before the A-frozen arm. I queued D4's pilots only at 07:21, so the A-frozen
same-lr cell (07:1x, T 208) and the A-frozen ladder lrs 10, 5, 2, 1 ran first. Order only: no rule, budget or fit row
changed, and both arms are exploratory. The D4 pilots run next, then its 4 sizes; the rest of the A-frozen ladder is cut (D5).
Vera #20357's extra cell (SGD lr 0.02 at alpha 1/64, seed 0, exploratory) is queued after the lr/2 control.
