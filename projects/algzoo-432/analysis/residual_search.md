# T16 — residual feature search on top of C1 (owner Iris, round 2)

Question: can 1-5 correction terms on top of C1 (2nd argmax with a phantom 0 before x_0, holdout 0.95503)
reach the noise-floor bar 0.98671? Answer: **no**. Every rule family we tried stays at or below 0.974,
even with 1e4-1e5 fitted constants. The negative result has three parts below.
All numbers: agreement with the model's argmax, N(0,1) inputs, train dev seed 81 (2e6), test dev seed 82
(1e6 or 3e5). The holdout 432432 was never used. Every constant here is FITTED to model outputs
(discovery only, PREREG 3(ii)). None of these is a candidate deliverable.

## 1. Candidate-set ceilings (oracle chooser): `residual_percfg.py` -> `residual_percfg.json`

Ranks in the phantom-augmented order (0 = max M, 1 = 2nd S, 2 = 3rd T, ...). An oracle picks the model's
answer when it is in the set, else S.

| oracle candidate set | agreement |
|---|---|
| {S} (= C1) | 0.95509 |
| {M, S} | 0.96392 |
| {S, T} | **0.98573 < bar 0.98671** |
| {S, T, 4th} | 0.98821 |
| {M, S, T} | 0.99443 |
| {M, S, T, 4th} | 0.99673 |

So a perfect S-vs-T near-tie corrector cannot pass. A passing rule must also predict when the model picks
the max (0.88% of inputs) or the 4th. Model max-picks sit at M-S < 2% of |M| (P(pick M) 0.30-0.38 at
gap < 0.5%, 0.003 at gap 4-8%), on both orders (S before or after M).

## 2. Half of C1's misses are robust, half are weight-noise fragile: `residual_fragile.py` -> `residual_fragile.json`

Fragile = the model's argmax changes under at least one of 20 draws of the bar's own noise (0.1%
multiplicative on all 432 weights). Mean model-vs-noisy-model agreement in this sample 0.9868 (= bar).

| | share of all inputs |
|---|---|
| fragile inputs | 5.97% |
| C1 misses on fragile inputs | 2.47% (C1 agrees 0.587 there; a noisy copy of the model agrees ~0.78) |
| C1 misses on robust inputs | 2.04% (C1 agrees 0.978 there) |

Robust misses: the model picks the 3rd in 74.5%, the max in 17.3%, the 4th or lower in 8.2%. Their
gap (S-T)/s has median 0.01, the same as the fragile misses. So a passing rule must reproduce ~2 pt of
systematic near-tie reversals AND match the model on fragile inputs better than C1 does.

## 3. Fitted rule families (all below the bar)

| family | fitted constants | test agreement | model-error-set agreement | file |
|---|---|---|---|---|
| C1 (no constants) | 0 | 0.95509 | 0.084 | algorithm_compact.py |
| top-3 candidates, score = linear functional of (x, relu x), indexed by candidate rank + position | 663 | 0.95556 | 0.083 | residual_ceiling.py (`--K 3 --feats x,relu`) |
| per-config (pM,pS,pT) linear S-vs-T comparator on x | 9,000 | 0.96569 | 0.417 | residual_percfg.py |
| same, bank x + relu(x) + running max + running 2nd max | 33,300 | 0.96825 | 0.451 | residual_percfg.py |
| per-config linear, candidates M/S/T, bank x | 17,100 | 0.96771 | 0.485 | residual_percfg.py |
| per-config linear, candidates M/S/T, full bank | 63,270 | **0.97062** | 0.527 | residual_percfg.py |
| candidate-centric MLP (46 features, 1x64) | 3,073 | 0.95748 | 0.128 | residual_mlp.py |
| candidate-centric MLP (46 features, 3x256), 15 epochs, still rising | 143,873 | 0.97400 | 0.597 | residual_mlp.py, scratch/iris_mlp_big.log |

Train and test agree within 0.002 for the per-config fits, so the gap is in the feature family, not in the
data. The 144k-parameter MLP was still improving (0.9735 -> 0.9740 in the last epoch), so 0.974 is not
its ceiling; it is still 1.3 pt under the bar with 700x the 200-literal cap.

## 4. Why linear correction terms fail: the comparator changes form inside one config

Exact Jacobian of logit[pS] - logit[pT] (finite differences, eps 1e-7, homogeneity check error 1e-5) on
near-ties (S-T < 0.1 rms) inside one config:

- config (pM,pS,pT) = (4,6,5), 597 near-ties: dominant patterns d = 80 S - 77 T - 2 M, |coef| <= 4 on the other inputs.
  11% of samples have an M coefficient of +-13..300 (e.g. 55 M + 19 S - 77 T, or 299 M - 254 S - 40 T).
  These are triple near-ties: when S follows M closely, n4 (new-max increment, ReLU(x - 0.852 n2 - 0.986 n4),
  subA.md) fires partly and M enters the S-vs-T comparison.
- config (5,0,9) (T is the last input), 594 near-ties: the median Jacobian classifies only 0.576 of them.
  Patterns include 105 S - 110 T, 75 S - 60 T, and 70 S - 35 M - 40 T: the fresh last input T is read
  through the deficit M - T (n7), and which deficit neurons are on depends on values.

So within one position configuration the S/T/M weights jump between several linear pieces, and the switch
depends on value comparisons against leaky registers. This is consistent with Felix's T15: rule-predicted
ReLU masks reach 0.9626 at best, with the B oscillator masks as the bottleneck.

## Verdict for RESULTS.md

No 1-5-term correction to C1 exists in the families tested. The best fitted rule-type predictor is
0.97062 (per-config piecewise linear, 63k constants) and the best black box 0.974 (144k). The bar needs
both near-tie reversals (2 pt robust) and max-picks; neither is a low-dimensional function of ranks,
gaps, positions, or linear crosstalk.

Reproduce (from swarm-algzoo-432/): `.venv/bin/python residual_percfg.py` (35 s),
`.venv/bin/python residual_fragile.py` (4 s), `.venv/bin/python residual_ceiling.py --K 3 --feats x,relu --ntr 1000000 --nte 300000 --epochs 5` (5 s),
`.venv/bin/python residual_mlp.py --hidden 256 --layers 3 --K 4 --epochs 15` (5 min).
