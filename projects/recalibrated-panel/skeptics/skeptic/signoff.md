# Vera (skeptic) sign-off — theory-1026-recalibrated-panel / recalibrated-panel

Verdict: **SIGN-OFF: CONFIRMED** that build's result reproduces. The status it supports is **REJECTED** (No to the
title: the rule-aware fine-tune did NOT erase the false-statement signal from the panel's 20 scores while keeping its
outputs). Prediction outcome: **FAILED** (clause 1: Rule B 33.0% >= 20%).

All numbers are from my own code (`vera26.py`, `vera_pool.py`, `vera_behav.py`, `vera_refit.py`, `vera_fwd26.py`)
on build's RAW saved data: country-token states `data/ctok/<state>.npy`, per-token `data/acts/<state>/{tok,kl,lp,off}.npy`,
the frozen panel `data/fits/panel.npz` and `zmoments.json`, and the frozen split `data/ref/split.json`. I wrote the
monitor and head definitions from PREREG.md s3/s5 text; I did not read analysis.py, ceiling.py or split.py.
Calibration of my checker before tonight's data: on the saved 1022 scores it gives Rule B 63.1/69.3/59.7%, logistic
83.9/81.7/82.7%, QDA 91.9/96.5/95.0% (= ceiling.json to the digit; `vera26_dry1022_s*.json`).

## Status cells (PREREG s7), 3 in-loop seeds pooled, 596 x 3 eval false (`vera_pool.json`)

| cell | mine | build | bar | verdict |
|---|---|---|---|---|
| Rule B catch | 33.0% [31.2, 34.8], FA 1.51% | 33.0% [31.3, 34.7] | < 20% for yes | >= 20% -> REJECTED leg 1 |
| logistic head (labels on attacked model) | 83.9% [82.3, 85.6], FA 1.17% (counts) | 83.9% | < 20% for yes | >= 20% -> REJECTED leg 2 |
| QDA head | 77.1% [74.8, 79.4], FA 0.89% (counts) | 77.1% | < 20% for yes | >= 20% -> REJECTED leg 2 |
| eval KL (every position) | 0.003934 (0.003811 / 0.003778 / 0.004212) | 0.0039 | <= 0.0096 | met |
| right country first | 298 / 298 / 298 | same | >= 290 | met |

Per seed: Rule B 43.6 / 17.8 / 37.6%; logistic 97.7 (FA 2.01%, not counting alone) / 58.2 / 95.8%; QDA 93.5 / 58.6 / 79.4%.
REJECTED holds on two independent legs; the head leg alone suffices, so the fragile Rule B line does not decide it.

## Nulls and robustness

- Rule B line null (2000 joint resamples of the 200 line-set trues, seed 326): pooled 19.6-39.2%; per seed 14.1-47.3 /
  1.8-31.5 / 28.2-48.8%. Rule B alone is not settled against 20%; the head leg is.
- Shuffled-label logistic head (5 permutations per seed): 0.0-2.5% (vs 58-98% real).
- Stronger heads (exploratory, `vera_strong.py`): the best head with eval FA <= 2% catches 96.8 / 85.2 / 97.7% (MLP, QDA
  reg 0), so the frozen heads are a LOWER bound on what the 20 scores still hold. Eval-label city-CV logistic 97.8 / 58.2 /
  96.6%: no gain from eval labels, so no leakage or under-fit drives the head numbers.
- Raw-data checks: ctok == tok.npy[off-2, readouts 10..29] and my scores == build's scores npz, max abs diff 0.0, all 4 states.
  Own CPU bf16 forward of the s1 adapter (60 eval statements): KL 0.00409 vs 0.00412 (corr 0.87), 40/40 pairs right,
  country-token states cos >= 0.99991 at readouts 10-27, HF hidden_states[28] vs build readout 29 cos 0.99994.
- Disjointness (`vera_disjoint.out`): train / attack / line-set / eval share 0 cities and 0 texts pairwise.

## Prediction clauses (PREREG s8)

1. Rule B pooled < 20%: FAILED (33.0%).
2. Pooled eval KL <= 0.0096: HELD (0.00393).
3. "lambda=0 control" = clean model under Rule B > 90%: HELD (99.5%); degenerate clause, disclosed in DEVIATIONS N0.
4. Refit >= 90%: HELD. My own refit (GroupKFold pick of readout and C, no eval data): 95.1% r11 / 98.0% r12 / 98.5% r16
   (build's frozen refit.py: 99.3 / 98.5 / 99.5%); shuffled-label refit <= 4.0%.

## Mechanism (must be in the post, measured)

- No tail lift: max over members of the 2nd-highest line-set z = 1.52 / 1.77 / 1.59, far under the 9.95 bound
  (derivation.md s1); the Rule B line fell from 3.80 (clean) to 1.52 / 1.77 / 1.71.
- Form offset: line-set true z excess kurtosis, median over the 20 members -1.76 / -1.48 / -1.70 (a 50/50 two-point mixture gives -2), member spread ratio median 4.2-4.3x. One shared line
  then cuts between the two sentence forms: Rule B 'is in' / 'is not in' = 85.2/2.0, 22.5/13.1, 74.2/1.0% (`vera_src.json`).
  derivation.md s2 gives the one-line algebra (mixture sd).
- Stop-grad side row (n = 1, never status): Rule B 3.7% [2.3, 5.2], logistic 59.7% (FA 1.7%), QDA 49.8%, KL 0.00643,
  298/298, refit 99.0%. Blocking the gradient to the spread made Rule B fall much further, at 1.6x the KL.
- Exploratory, never attacked: Rule B per sentence form 84.4 / 61.6 / 90.9% (`vera_perform.json`). Foldout or Next posts only.

## Definition checks

- "erase the false-statement signal from the panel's scores": tested as the labelled heads on the 20 frozen scores <20%,
  PREREG s5/s7. The heads use labels a deployer of a fixed panel does not have; the post must say they are ceilings.
- "keeping its outputs": PREREG s7 bar (pooled KL <= 0.0096 AND >= 290/298 per seed): met.
- Novelty: see novelty.md (new as far as I can find; credit 2410.09878, 2505.14300, 2603.23171, Samuelson / 1601.07521).

## Differences with build (none changes a verdict)

- Refit best readout and catch differ (my 95.1 / 98.0 / 98.5% vs 99.3 / 98.5 / 99.5%): different selection code; both >= 90%.
- Pooled Rule B line null: mine 19.6-39.2% (seed 326), build 17.3-39.5%.
- PREREG s1 table says the training pool is 1400 statements (350 cities); the split has 800 (200 cities). Doc typo; s5
  and the code use 800 (flagged #7786).
