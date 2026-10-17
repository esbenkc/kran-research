# T15 — pattern-predicted linear model (Felix, round 2)

Idea (Magnus #1868, track 2): the RNN is piecewise linear. Given the ReLU pattern (which neurons are on
at h_1..h_10), the logits are an exact linear function of x. If simple rank/gate rules predicted the
pattern, the algorithm would be "pick the pattern by rules, then apply a linear map".

All numbers: dev seeds only (train 41, test 42, 2e5–3e5 samples), agreement with the model's argmax.
No holdout run (nothing here is a candidate). Forced-mask forward: h_{t+1} = m_{t+1} ⊙ (W_hh h_t + W_hi x_t)
with the exact 432 weights; only the masks m are replaced.

## Result

**Negative.** Kasper's split (#1877): true-pattern oracle = 1.0000 (by construction, checked
`assert` in the scripts). Rule-predicted pattern, best we found = **0.9626**, with ~160 depth-10 trees
on value features (tens of thousands of leaves) and the exact weights downstream. That is far from a
compact rule and still below the floor 0.98671. Rank-order rules alone give 0.938, below the null 0.953.

| pattern source (exact weights downstream) | all masks | only A {1,2,4,6,7,8} | only gates {0,5,9} | only B {3,10–15} |
|---|---|---|---|---|
| true pattern (oracle) | 1.0000 | — | — | — |
| ORDER of prefix {0, x_0..x_t}, depth-12 tree per (neuron, t) | 0.9380 | 0.9435 | 0.9844 | 0.9791 |
| 10 streaming VALUES (x_t, x_{t−1}, x_{t−2}, M, S, leaky max 0.969, t−pM, t−pS, M−x_t, S−x_t), depth-10 tree per (neuron, t) | **0.9626** | 0.9874 | 0.9911 | 0.9741 |

Single-neuron mask replaced (others true), from `pl_oracle.py` and `pl_rank_tree.py`:

| neuron | active rate | always-on | per-(t) majority | order-tree |
|---|---|---|---|---|
| n0 | 0.75 | 0.766 | 0.766 | 0.997 |
| n1 | 0.73 | 0.326 | 0.326 | 0.987 |
| n2 | 0.90 | 0.999 | 0.999 | 1.000 |
| n3 | 0.77 | 0.978 | 0.978 | 0.994 |
| n4 | 0.23 | 0.042 | 0.273 | 0.994 |
| n5 | 0.41 | 0.588 | 0.415 | 0.991 |
| n6 | 0.72 | 0.198 | 0.254 | 0.972 |
| n7 | 0.77 | 0.163 | 0.163 | 0.983 |
| n8 | 0.69 | 0.439 | 0.472 | 0.984 |
| n9 | 0.26 | 0.116 | 0.684 | 0.995 |
| n10 | 0.89 | 0.980 | 0.980 | 0.995 |
| n11 | 0.88 | 0.986 | 0.987 | 0.998 |
| n12 | 0.90 | 0.996 | 0.996 | 0.998 |
| n13 | 0.90 | 0.999 | 0.999 | 0.999 |
| n14 | 0.86 | 0.988 | 0.988 | 0.997 |
| n15 | 0.73 | 0.917 | 0.962 | 0.991 |

Second ceiling, the other half of the idea (logits linear in x within a rank class). One least-squares
linear map x → logits per class, fit on the model's logits (fitted, discovery only):

| class key | classes | fitted coefficients | agreement |
|---|---|---|---|
| (pM, pS), phantom = −1 | 106 | 10,600 | 0.9572 |
| (pM, pS, pT) | 871 | 87,100 | 0.9592 |

## Reading

- The A masks (max bookkeeping, deficits) are **not rank predicates**: from order alone the n6/n7/n8/n1
  masks are 2–3% wrong and each costs 1.3–2.8 pt. With values (M, S, leaky max, lags) A masks become
  good enough (0.987 alone): A thresholds compare values, not ranks.
- The B masks (oscillator bank) are the bottleneck once values are allowed (0.974 alone). Ages t−pM,
  t−pS are in the feature set, so a clock phase that is a function of "steps since max/2nd" does not
  predict where B neurons switch off. B switching depends on finer value history.
- Logits are not linear within a rank class: 87k fitted coefficients give 0.959, the same band as all
  the rank/gap ceilings (0.955–0.965).
- Consequence for round 2: the pattern-linear route cannot give a compact algorithm at the floor. Its
  rule half needs tens of thousands of tree leaves for 0.963, and its linear half needs the exact
  per-pattern W products (the weights themselves, flagged by PREREG item 4).

## Reproduce (from swarm-algzoo-432/scratch, venv ../.venv)

- `../.venv/bin/python pl_oracle.py` (5 s): oracle = 1.0 assert, always-on / majority per neuron.
- `../.venv/bin/python pl_rank_tree.py 12` (22 s): order-feature trees.
- `../.venv/bin/python pl_value_tree.py` (4.5 min): value-feature trees + per-class linear maps.
