# skeptic3 sign-off — grokked-llc-closed-form (2026-10-15)

**Decision: SIGN-OFF on the NULL claim**, with the scope below.

Claim signed: *At the pre-registered, frozen SGLD setting (gamma = 1, n·beta = 1e4, calibrated to within 3.5% on 8
networks with a known LLC), the estimated LLC of grokked quadratic networks on modular addition is 65-78% of the
closed form K_eff(3p-1)/2 of Cullen, Estan-Ruiz, Danait and Li (arXiv 2603.01192 v3, Remark 5.5), in every
inside-hypothesis configuration from p = 11 to p = 23 (5 configurations, 12 networks). 0/5 configurations fall in the
pre-registered ±15% band, all ratios are below 1, so by plan.md the verdict is NULL.*

Not signed: any claim that the paper's formula is "wrong" as an asymptotic statement (except the exact shift-symmetry
correction below), and any claim about p = 31.

## 1. Independent derivation (derivation.md, sympy_neuron.py, rank_check.py, fourier_rank.py)

* Generic count per neuron on concatenated one-hot inputs is 3p-2, not 3p-1: the shift (u, s) -> (u + c1, s - c1)
  leaves u[a] + s[b] unchanged. Exact sympy rank at a rational point: p = 3 -> 7, p = 5 -> 13; generic inputs give
  3p-1. So the generic value on this data is K_eff(3p-2)/2, at most 1/(3p-1) <= 5% below the paper's.
* Cap: rank J <= p(N-1), so the formula can hold only while K_eff(3p-2) < p(N-1). Numerically rank J =
  min{K(3p-2), p(N-1)} in 15/16 random cases (one near-cap defect). The 'inside_hypothesis' flag in llc.csv equals
  this condition on every grokked row (asserted in verdict_recompute.py).
* At Fourier points, neurons that share a frequency span at most 20p-36 directions (my count; skeptic2 found
  F(20p-36) - 16·C(F,2) with an independent phase-grid construction). Trained checkpoints are not exactly pure, so
  their Jacobian is full rank but with a very flat tail. This explains part of the gap; it is a numeric lead, not a
  proof, and the post presents it as such ([^pure]).
* I checked every formula in the post against this: lambda_grok, lambda_mem (frozen first layer, linear in V,
  exact), the 3p-2 footnote, the K_eff(3p-2) < p(N-1) condition, (p-1)/2·(16p-24) = 3784 and K(3p-1) = 6732 at
  p = 23, K = 99, and lambda_hat = n·beta(E[L] - L(w*)).

* Lower bound used in the post ("which number is off"): at a zero-loss point, K(theta) = 1/2 |J delta|^2 + O(|delta|^3);
  by the splitting lemma lambda = rank(J)/2 + (RLCT of the remainder in the other coordinates) >= rank(J)/2. For the
  12 verdict networks llc.csv has rank J = K_eff(3p-2) exactly at p <= 23 (SVD gap 1e7-1e8 at p <= 17; p23_K92 Gram
  method gives 6162-6164 of 6164), so rank(J)/2 is 0.969-0.985 of the formula, and SGLD reads 0.67-0.79 of rank(J)/2.
  So the reading sits below a hard lower bound on the asymptotic LLC: the gap is a property of the finite-resolution
  estimate, not of the formula (beyond the exact shift correction). Caveat (skeptic2 #674): after polishing, the last
  directions are full rank only at relative SVD tolerance ~1e-8..1e-10, so the bound only shows at n·beta far beyond
  any usable sampler. The post must keep the "infinitely fine sampler" wording and not say "the true LLC is lower".

## 2. Re-run from saved raw data with my own code (sgld.py, rerun.py; no build code imported)

Own numpy SGLD (Lau et al. 2308.12108 form), gamma = 1, eps = 0.1/(n·beta·h_max), 20k steps, 5k burn, 3 chains,
raw checkpoints (no polish). Validated on 5 planted smooth teachers: within 1.2% of rank/2 at n·beta 1e3/1e4/1e5
(validate_sgld.json).

Build lambda_hat vs mine at n·beta 1e4 (verdict_recompute.json, `crosscheck_build_vs_skeptic3_nbeta1e4`; mine = mean
of two chain-seed sets): **all 12 networks of the 5 verdict cells (p11_K33 x3, p17_K51 x2, p17_K68 x3, p23_K69 x1,
p23_K92 x3) agree within 1.2%** (max |rel| 0.0118, p23_K92_s2). Outside-hypothesis cells agree within 6% (p11_K88:
-5.5% / -6.0%, mine lower). p31 frac 0.7 (llc_p31_frac07.csv): p31_K124_s0 build 0.757 vs mine 0.752 of the formula.
Per-checkpoint table: rerun_summary.json (summarise_reruns.py).

## 3. Verdict recomputed from raw llc.csv (verdict_recompute.py -> verdict_recompute.json)

My code recomputes lambda_paper and the inside-hypothesis flag from p, K_eff, l_rank, N (asserted equal to the CSV),
takes the median over seeds per (p, K) cell, and applies the plan.md rule. The verdict does not depend on how p = 31
or unsettled cells are treated:

| exclusion choice | cells in ±15% band | log-log slope | verdict |
|---|---|---|---|
| all inside cells (incl. p31 frac 0.5) | 1/7 | 1.08 | NULL |
| drop p31 frac 0.5 | 0/5 | 0.97 | NULL |
| also drop unsettled (stationarity.csv) | 0/5 | 0.97 | NULL |
| only cells with a settled 40k-step check | 0/5 | 0.99 | NULL |

Per-cell ratio (p11-23): p11_K33 0.721, p17_K51 0.745, p17_K68 0.655, p23_K69 0.782, p23_K92 0.675. All < 1.
Against the shift-corrected value K_eff(3p-2)/2 the ratios rise by at most 5% and stay outside the band.
Memorising (V-only, phase label passes both predictors, n = 11): ratio 0.93-1.02, median 0.997 — a sampler check,
because that model is linear in V.

## 4. Null tests (mine: chain seeds + n·beta sweep)

* Chain-seed sets 0 vs 1 at n·beta 1e4: differ by <= 1.3% on every grokked checkpoint with p <= 23 (<= 1.8% at p31).
  Sampler noise is not the gap.
* n·beta sweep 1e3 / 1e4 / 1e5 (rerun_summary.json): all 12 verdict networks move **+1.2% to +7.4%** from 1e4 to
  1e5; at 1e5 their ratios to the formula are still **0.68-0.79**. The NULL survives a 10x finer resolution.
* Non-monotone n·beta dependence is not by itself proof of non-equilibrium: p11_K33 reads 0.77 / 0.72 / 0.74 at
  1e3 / 1e4 / 1e5 and is stationary at 40k steps (lead's stationarity.csv). I withdraw that part of my room post #705.
  The 40k-step stationarity check is the right inclusion criterion.
* p31: frac 0.5 K93 is 0.41 / 0.93-0.96 / 0.78 and keeps drifting at 40k steps (lead): correctly excluded.
  frac 0.7 K124 (s0-s2): 0.385 / 0.74-0.75 / 0.70; frac 0.7 K93 s0: 0.414 / 0.860 / 0.787 (build 0.873 at 1e4).
  The only inside cells that DROP from 1e4 to 1e5 are at p = 31 (-5% to -9%); at 1e5 p31 reads 0.70-0.79, the same
  band as p11-23 at 1e5 (0.68-0.79). So p31's higher 1e4 reading (0.86-0.87 at K93) is not stable across sampler
  settings and must not be read as "the gap shrinks with p". Whatever is done with p31, it cannot change the verdict:
  best case 2/7 cells in band (< 80%).

## 5. Novelty (novelty.md)

No prior estimated-vs-closed-form test for this model found. The paper itself (v3) only checks scaling and reads
absolute LLCs "up to an estimator-dependent scale", so the post must (and does) frame the NULL as being about the
absolute number at a calibrated setting, not about the scaling claim. Caveat: citation graph of 2603.01192 not checked
(Semantic Scholar rate limit).

## 6. Post checks (draft v5 -> v7, all raised in the room and applied by write)

Checked every number in v7 against llc.csv, calibration_frozen.json, stationarity.csv, my reruns and my derivation.
Fixes I raised that are now in the text:
1. Memorising points described as frozen-first-layer, output-weights-only: a sampler check, not comparable with the
   grokked value.
2. Range 0.65-0.78 (not 0.66); polish "at most 3% in grokked networks"; training loss "below 4e-7".
3. n·beta: "Ten times finer moves it up by at most 7%" (p23_K69 +1.2%, p17_K68 +7.4%); scoped to p = 11-23 if p31
   enters, with "At p = 31 it moves down 5-7%".
4. Rank tolerance: 1e-9 (SVD) for p <= 17, 1e-6 (Gram route) for p = 23.
5. "Each further factor of ten needs ten times more steps" marked as following from our step-size rule, not measured.
6. Lower-bound section: the bound and "LLC = rank/2 at a max-rank point" are statements about the LLC as a limit; the
   sampler at finite temperature reads below them. Formula right "up to one direction per neuron".

Remaining condition for my sign-off: the figure, summary.json and the post use the same cell set (p31 frac 0.5 out;
p31 frac 0.7 in only if its 40k-step check settles; if in, the text drops "every size / constant ratio" and gives the
range 0.65-0.87). The verdict is NULL in every variant.
