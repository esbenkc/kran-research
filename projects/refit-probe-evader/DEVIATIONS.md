# Deviations and notes after PREREG.md

Each entry: the Copenhagen time of the commit that first recorded it (`TZ=Europe/Copenhagen git log`), and whether
any scored output had been read at that time. The rule in PREREG.md is still reported.

PREREG v1 = 8467ee5 (15:04:13 CET). v1.1 = 5df39fa (15:30:31 CET). R1 step 1 at 07:32 PDT = 15:32 CET.

## N1 — naming of the chosen LEACE setting (recorded during R1, before any R1 output was read)

PREREG s4 calls the chosen setting "Ledoit-Wolf shrinkage". The code (`leace_price.py:48`) uses
`LeaceFitter()` with the library defaults of concept-erasure 0.2.4: `shrinkage=True`, which is an optimal linear
shrinkage toward tr(S)/p times the identity with coefficients from arXiv 1308.2608, not Ledoit-Wolf (Vera #8423;
her numpy transcription matches the library to 7.6e-16, `skeptics/skeptic/vera_shrink_check.out`). The post says
"linear shrinkage (the concept-erasure default)". The number P = 0.006145 is unchanged.

## N2 — the registered eraser keeps the library's trace constraint (recorded during R1, before any R1 output was read)

The library default `constrain_cov_trace=True` mixes the oblique LEACE projector with an orthogonal one when
erasure would raise tr(cov). The registered eraser was fit with that default (`leace_price.py:145`, other arguments
at the defaults), so P is the price of the library's default eraser, which may not equalise class means exactly.
P stays the registered number. Vera rebuilds P both ways (pure LEACE + shrinkage; library default); both are
reported beside P, with no status role. If the pure-LEACE price falls outside [P/3, 3P] while P is inside (or the
reverse), the post says so next to the claim.

## D1 — mean-gap arm G: loss summed over the whitened dimensions, not averaged (recorded before G ran)

PREREG s2 writes G's loss term as "mean over R of mean(d_R^2)". Averaging over the 1536 whitened dimensions gives a
term of ~0.10 per readout at step 1 (Bruno #8424, `data/logs/smoke_meangap.log`), against lam x KL ~ 30-90, so at
lam 1e4 the arm is close to a KL-only run and cannot test the stronger attacker it was planned as. The intended
quantity is the squared whitened mean gap ||W_R (mean_false - mean_true)||^2 (Felix #8339, Bruno #8332), i.e. the
SUM over dimensions (~150 at step 1, the scale of the refit50 loss, ~30-65). G uses the sum. G is a side arm with no
status role; no G output existed when this was written.

## D2 — the goal's refit-interval arms restored as slots 7-8 (recorded before R1 was scored)

The plan cut the goal's refit-every-25 and refit-every-100 arms for time (PREREG s2). The smoke measured 2.2 s/step
and ~18 min per run with extraction (Bruno #8424), not the ~38 min the plan assumed, so 6 runs end near 09:50 PDT.
Slots 7 and 8 are added, each only if it can start by 10:15 PDT (18:15 CET): slot 7 = retrained-probe fine-tune with
refit every 25 steps (steps 0, 25, ..., 275), lam 1e4, seed 0; slot 8 = the same with refit every 100 steps (steps 0,
100, 200), lam 1e4, seed 0. Both are n=1 descriptive rows (refit interval vs catch and output change); neither enters
the status or the prediction outcome.

## D3 — shuffled-label arm N: probe term scaled to the base arm's starting loss (recorded before N ran)

As frozen, N's starting probe loss (mean over readouts 10..29 of the mean squared twin gap of probes fit on
permuted labels) is 1.74 against 77.8 for true labels: 2.2% (Iris #8430, `research_checks/null_gap.py`,
`null_gap.json`, clean fp32 states, the exact `attack_refit` fit). The strong C at deep readouts keeps the
random-label probe near flat. N's gradient would then be ~7x smaller and N would pay a small KL by loss scale alone,
so clause 5 (N's KL < 0.5x the base KL) would pass without testing whether the price is LoRA machinery. N's probe
term is multiplied by the fixed constant c = 77.8 / 1.74 = 44.7 (true-label over shuffled-label starting loss on
the clean model), so N starts with the base arm's loss scale. This makes clause 5 harder to pass, not easier. Same
reason as D1 for G (Tove #8425). No N output existed when this was written.

D3 diagnostic (Kasper #8431, recorded with D3, before N ran): report the fraction of the step-0 probe loss removed
by step 300 (from the per-step `probe_loss` log), base seeds vs N. If N removes less than half the base arm's mean
fraction, clause 5 is "not comparable" and is treated like "N did not run" in PREREG s6 (no CONFIRMED on it).

D3 implementation (Bruno #8438, recorded before N ran): the factor is measured per run, not the fixed 44.7:
k = R1's logged step-1 probe loss / N's own step-1 probe loss (same seed, so the same first twin batch), frozen for
the whole run and saved as `k_match` (`attack_refit.py --mode shuffled --match data/runs/refit50__lam10000__seed0.json`,
run `shuffled__lam10000__seed0__matched`). This is the form Kasper #8431 and Tove #8433 asked for; Iris's and Vera's
clean-model ratio 44.6 (permutations 1.49-2.22 -> about +-25%) is the expected value. The logged N probe loss is the
scaled one (raw = logged / k). Only the matched N runs.
