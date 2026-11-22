role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: cue-size-law-8b-rep, continuation room theory-1122-cue-size-law-8b

**Sign-off: CONFIRMED.** The frozen rule (PREREG.md, FROZEN RULE block, unchanged since the pushed prereg commit
ec16e6569 / header a4215318a) gives yes on tonight's branch, and every number it needs reproduces from the raw run
JSONs with my own code. The Nov 20 sign-off (UNCONFIRMED, C1 seed 6 had no raw data in git) is kept at
../2026-11-20/skeptic/signoff.md.

## What changed since Nov 20
- labq 9y8gn3qvoc (DONE, d5e6bfb8f) recovered data/runs/l8_c1_s6.json from the failed job's work dir.
  Its test gap 0.014066543337612529 equals, to the last digit, the log line written before the job failed
  ('10:05:25 [B] OK l8_c1_s6 DONE ... test_gap 0.014066543337612529'), so the run finished before the frozen
  11:00 PST wrapup (not 'incomplete' under REP-4); only its commit was late (vera_c1s6_provenance.txt).
- The mini rerun M2 was stopped at update 2,624 and never wrote a JSON, so provenance() takes the laptop copy.
  The M2 log equals the laptop file bit for bit at 47 of 47 shared evals (vera_c1s6_provenance.txt).

## Rebuild (vera_check.py on dev_curve + dense_curve, gap recomputed from px_off/px_on)
| claim | my number | build (data/analysis.json @ 9e14dbf2f) | match |
|---|---|---|---|
| dense T s5 / s6 / s7 | 24/40/72/152/296/640; 24/40/72/136/280/688; 24/40/64/120/304/640 | same | 18/18 |
| per-seed slopes | -0.9550 / -0.9585 / -0.9534 | same | yes |
| seed-level 95% (t, df 2) | -0.9556 [-0.9620, -0.9492], x1.94 [x1.93, x1.95] | same | yes |
| prompt-level (2,000 x 64 dev, seed 0) | [-0.9734, -0.9545] | same | yes |
| censored / Tobit | 0 / -0.9556 | 0 / -0.9556 | yes |
| C1 s5 (mini) | 8,192/8,192, no switch, max D 0.0296 | same | yes |
| C1 s6 (laptop) | 8,192/8,192, no switch, max D 0.0327 (dense curve; 0.0269 on the 64-grid only) | same | yes |
| C2 s5, s6 (cue 0) | max abs gap 0.0 over 1,024 updates each | same | yes |
| G1, G2 (vera_gates_1122.txt) | G1 dev mean 0.3435; G2(b) pass; G2(a) whole-prompt pass | pass | yes |
| status | CONFIRMED | CONFIRMED | yes |
Field-by-field diff: 70 fields, 0 mismatches (vera_diff.py, vera_diff.txt). M3 left status_8b byte-identical.

## Null tests
- Shuffled-coin control (the frozen null): 2 of 2 seeds never switch in 8,192 updates (max D 0.030, 0.033 vs 0.10).
- Within-seed size-label shuffles (20,000, numpy seed 0): the seed-level interval lands inside [-1.4, -0.7] in 1 of
  20,000; mean slope under the null 0.00 [-0.47, 0.48] (vera_seed_null.txt, check_final.json).
- Luck check on the narrow interval: all 3 seeds share cue direction u1110 and one cache, and tie at T = 24 and 40 at
  the two largest cues. With the 8B Nov 16 seed sd 0.0314 the 3-seed interval would be [-1.034, -0.878]; with the
  0.6B sd 0.0395 it would be [-1.054, -0.857]. Both inside the band.
- Leave-one-size-out mean slope: -1.015 to -0.898, all inside the band.

## Side test S9 (fresh cue directions; exploratory, never in the status)
- Rebuilt from raw curves with the frozen |gap(0)| >= 0.02 exclusion from data/scan_u*.json (vera_s9.txt):
  u1125 s5 -0.970 (k0-k5), u1126 s6 -1.074 (k2-k5), u1127 s7 -1.035 (k1-k5); n = 3: -1.026 [-1.157, -0.895],
  x2.04 [x1.86, x2.23]. Inside the band. Matches analysis.json secondary.S9_fresh_u.
- The S9 seeds drop the largest sizes, where the curve is shallower. On the same sizes the main seeds give -1.085
  [-1.195, -0.974] (k2-k5) and -1.015 [-1.041, -0.989] (k1-k5) (vera_matched.txt), so the steeper S9 mean is mostly
  the size subset, not the direction.
- The one mini cell m8u1127_k5_s7 (T 664) equals the partial laptop log of the same cell, in git before M3, at 15 of
  15 shared evals (vera_m3.txt).

## Definitions and novelty
- No stretched definition: share = norm share after block-34 RMSNorm (as frozen), switch = dense D >= 0.10 at two
  consecutive evals, band [-1.4, -0.7] (x1.625 to x2.639 exactly, ../2026-11-20 vera_band_sympy.txt).
- C2 s6 was added to meet the 2-seeds-per-control lint. With cue 0 the cue-on and cue-off inputs are identical, so
  max |gap| = 0 holds for any seed; it is a pipeline check, not evidence like C1.
- Novelty refresh (novelty.md): nothing new since Nov 20 measures RL updates-to-switch against cue size. The
  exponent was first published by the Lab as EXPLORATORY (Nov 16); this run is its pre-registered replication.
