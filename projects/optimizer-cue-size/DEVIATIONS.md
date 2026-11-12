# DEVIATIONS: optimizer-cue-size

Changes to the frozen plan (PREREG.md v1 84ee6d172, v1.1 8a7288ea2). Times are PST (CET - 9 h).

## D0: smoke runs before the frozen T4 slopes (Bruno #20309, Kasper #20306)

05:59-06:01 PST: three 64-update smoke runs at x 33.3 (alpha 1/32), fp32 cue, seed 0, to time the code and test the new
flags: AdamW 1e-4 (T 32), sign-SGD 1e-4 (T on all evals 40), SGD lr 0.1 (no switch; P(X | cue off) fell to 0.000,
first-token entropy to 0.00 and KL to 0.57 by update 16). The SGD smoke ran before Felix posted the toy slopes
(PREREG s7 says frozen predictions come before the first SGD update). One size and no switch carry no slope
information; nobody who made a prediction read it (Kasper #20306). The smoke run JSONs were deleted by mistake when the
folder was cleaned; the printed eval rows stay in `logs/smoke_{adamw,sgd,sign}.log`. Smoke runs never enter any fit.

## D1: text-health test recalibrated before any SGD pilot (Vera #20315, Kasper #20318; Felix, PREREG owner)

Written at 06:08 PST before any SGD pilot ran. The argmax-share test of PREREG s8.1 ('argmax share <= step-0 share +
0.25', Bruno #20292) fails AdamW itself: p0cast_a0.03125_s0 cue-off argmax share 0.672 at step 0, 0.9375 at step 25.
It is DROPPED. Text health (s8.1 lr rule and the s8.4 stop in every run) is now: dev cue-off first-token entropy >= 0.5x
its step-0 value AND dev cue-off top-1 mass <= 0.9 at every check. AdamW passes both with margin: p0cast s0 min entropy
ratio 0.83, max top-1 0.77 (12 checks); p0cast s1 and adamw fp32 s0 numbers in the room post. All other s8.1 tests
(finite loss, mean KL <= 0.5, P(X | cue off) in [0.07, 0.48] after update 8) stay.
