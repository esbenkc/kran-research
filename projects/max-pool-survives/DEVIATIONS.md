# Deviations — max-pool-survives

Every change after the PREREG freeze (commit 9bc1a08, 2026-10-19 15:05:47 CEST). Times from `git log`.

## D0. Re-run instead of reuse (known before the freeze; in PREREG.md)
The 1018 run saved no adapter, so lambda = 0 seeds 0-2 are re-run with the frozen recipe.

## D1. Amendment 1 to PREREG.md (commit 4bc18f2, 15:06:07 CEST; before any attacked data was scored)
Kasper #3183, posted at the freeze, reviewers Tove #3188 and Kasper #3187 asked for it:
(a) shuffled-label null = within-pair label swap on the probe split, refit max14;
(b) framing rule: "max pooling helps" only if the pooled paired max14 - last14 lower 95% bound > 0, else the
post says the answer tokens carry the signal;
(c) sigma1/sigma0 per monitor next to each bell-curve prediction.

## D2. Scored checkpoint named (Vera #3185; committed with this file, before any attacked scoring)
Only the final step 150 of each seed is scored. No intermediate check is used for any number in the post.
