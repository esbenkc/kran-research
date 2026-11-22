role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: cue-size-law-8b-rep, room theory-1120-cue-size-law-8b-rep

**Sign-off: UNCONFIRMED** (the frozen rule's answer at the wrapup, 20:00 CET 2026-11-20).
The numbers the build reports reproduce exactly from the raw curves with my own code. The status is not settled
only because the raw data of control C1 seed 6 is not in the repository: it finished on the laptop (log line
'10:05:25 [B] OK l8_c1_s6 DONE T None test_gap 0.014066543337612529 train_s 3562.3') but labq job 5f3rwdo6um
failed on a lost heartbeat at 10:27 PST before it committed any output. A log line is not raw data, so I do not
count it. Recovery: labq 9y8gn3qvoc (copies the finished JSON from the kept laptop work dir) and/or a mini rerun
m8_c1_s6 (the parent showed mini and laptop copies identical in 1,539 of 1,539 eval points, so the rerun must
reproduce test_gap 0.014066543337612529).

## What I rebuilt (vera_check.py, from dev_curve + dense_curve px_off/px_on and the per-prompt p_off/p_on)

| claim | cells | my number | build's analysis.json | match |
|---|---|---|---|---|
| dense T, seed 5, k0..k5 | m8_k*_s5 | 24/40/72/152/296/640 | T_dense same | 6/6 |
| dense T, seed 6 | m8_k*_s6 | 24/40/72/136/280/688 | same | 6/6 |
| dense T, seed 7 | m8_k*_s7 | 24/40/64/120/304/640 | same | 6/6 |
| per-seed slopes | 18 runs | -0.9550 / -0.9585 / -0.9534 | same | yes |
| seed-level 95% (t, df 2) | 3 seeds | -0.9556 [-0.9620, -0.9492] | -0.95563 [-0.96203, -0.94923] | to 1e-8 |
| prompt-level 95% (2,000 x 64 dev prompts, seed 0) | 18 runs | [-0.9734, -0.9545] | [-0.97342, -0.95453] | to 1e-8 |
| pooled Tobit (0 censored) | 18 runs | -0.9556 | -0.95563 | yes |
| inside band [-1.4, -0.7] | | yes | | |
| C1 s5 (mini, shuffled coin) | m8_c1_s5 | full 8,192, no switch, max D 0.0296 | | |
| C1 s6 (laptop) | l8_c1_s6 | NO RAW DATA in git | | blocks |
| C2 s5 (cue 0) | m8_c2_s5 | max abs gap 0.0 over 1,024 | | |
| G1 | gates.json | dev mean 0.3435 in [0.30, 0.50]; think top-1 0 | pass | yes |
| G2(b) REP-1 (vera_gates.py) | 64 dev | max 0.1287 <= 2 x 0.1396; median 0.0043 <= 0.01; top-1 63 strict, 64 with 1 NDIF tie | pass | yes |
| G2(a) (vera_g2a.py) | 64 dev | whole prompt max 0.00446 < 0.01 (pass); per token median 0.011, rows 1-2 0.0131 / 0.0108 | pass | caveat |
| cache determinism | mini cache | array sha256 70c4703a... = parent | same | yes |
| S8 (vs parent 0.6B, 5 seeds) | | G ratio 0.860; k2-k5 median ratios 0.900/0.895/0.974/0.941; holds | 0.8600 | yes |
| S10 (mean slope in [-1.003, -0.803]) | | -0.956, holds | | |
| S4 halves | | -0.764 (k0-k2) vs -1.138 (k3-k5), diff 0.374 < 0.5 | | |
| S2 calibrated constant | | median ratios 0.89/0.86/0.87/1.01/1.17, all within 2x | | |

Per-halving factor: 2^0.9556 = 1.94 (vera_band_sympy.txt); 0.6B reference 2^0.899 = 1.87.

## Null tests
- Shuffled-coin control C1 s5 (the frozen null): never switches in 8,192 updates (max D 0.030 vs 0.10).
- Size-label permutation within each seed (20,000 draws): mean slope 0.00 [-0.47, 0.48]; observed -0.956, p < 5e-5.
- Leave-one-size-out mean slope: -1.015 (drop k0) to -0.898 (drop k5); every value inside the band.
- With the parent's seed sd 0.031 instead of this run's 0.0026, the n=3 interval would be about -0.956 +- 0.077,
  still inside the band; the narrow interval is partly luck (all 3 seeds hit T = 24 at k0).

## Definitions checked (novelty.md)
- G2(a) holds only on the whole-prompt unit the parent coded before the freeze; per row the cue rows 1-2 exceed
  1e-2. Disclosed in the post (Selma, footnote). Not a post-data stretch.
- Bootstrap upper end equals the point because T lives on an 8-update grid (vera_boot_diag.py: 50-100% of
  resamples reproduce each T exactly). Not a bug.
- Novelty: no outside paper measures RL updates-to-switch against a planted cue's norm share across sizes; the
  closest theory (arXiv 2609.36117, supervised poisoning) implies slopes -2 or -4. The exponent itself was already
  published by the Lab as EXPLORATORY; this run is its pre-registered replication.
