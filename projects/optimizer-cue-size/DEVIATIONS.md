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
