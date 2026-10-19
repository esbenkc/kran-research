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

## D3. Finite-sample bell-curve value reported next to the frozen formula (Iris #3193, before attacked scoring)
The frozen formula Phi(D - (1-c)p - z) with z = Phi^-1(1 - 2/201) overstates the expected catch of the real
2nd-of-200 rule by up to 0.036 near the 50% bar. The hollow mark in the figure stays the frozen formula.
results.json also reports the exact finite-rule value E[catch] = int Phi(D - (1-c)p - t) f(t) dt, with f the
density of the 2nd-highest of 200 standard normals (theory/check_equations.py, catch_exact). Extra, not a
replacement; it enters no rule.

## D4. First scoring pass used the wrong CV split for three probes (Bruno #3206; fixed before any number was reported)
Bruno's first pass of analysis.py fit `mean8`, `fresh` and `last14` with the 1018 `prep.fit_probe`
(StratifiedKFold, which splits a city's statements across folds), not the frozen GroupKFold by city.
He switched to GroupKFold before reporting. The other version is kept: data/results_cv_stratified/results.json
(fresh 12.6%, mean8 0.3%, max14 0.0% in both; max14 used GroupKFold from the start). The status (REJECTED)
and the pre-registration outcome (failed) are the same in both versions. The post reports the fresh-probe
number's sensitivity to this choice (36.1% vs 12.6%).
