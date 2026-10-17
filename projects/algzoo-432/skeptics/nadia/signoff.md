# Nadia (skeptic3) — independent rebuild, algzoo-432

Date: 2026-10-17. Own numpy forward from the raw `m16_10.pth` state dict (no `core.py`, no `check_agreement.py`,
no other skeptic's code). Input seeds 4320xx only; the 432432 holdout was never used. Code and JSON outputs are
in this folder (copies of `swarm-algzoo-432/skeptic3/`). Run from `swarm-algzoo-432/skeptic3/` with
`../.venv/bin/python <script>`.

Verdict: **CONFIRMED** as "no description of at most 200 numbers found that reaches the 0.1% noise-floor bar".
One post sentence was false (0.3% noise level, item 3.1); it is fixed in the text Selma pushes.

## 1. Rebuilt numbers

| claim (RESULTS.md / post) | claimed | mine | where |
|---|---|---|---|
| forward = torch `nn.RNN(relu, no bias)` + `Linear` | — | argmax agree 1.0 (5e4) | nadia_check.py:10-19 |
| model accuracy = agreement of true 2nd argmax | 0.95283 (holdout) | 0.95324 (seed 432010, 1e6) | nadia_check.json `model_acc` |
| errors pick the 3rd / the max | 63% / 20% | 63.3% / 20.3% | `err_pick_3rd`, `err_pick_max` |
| C1 (rebuilt from its one-line text only) | 0.95503 | 0.95527 (432010), 0.95522 (432060) | nadia_check.py:33-37 |
| <=1 positive input: share / error rate | 78% wrong | 1.06% of inputs, 78.1% wrong | nadia_check3.json |
| bar: 0.1% multiplicative noise, 20 draws | 0.98671 | 0.98686 (sd 0.0036), 0.98697 (sd 0.0046) | nadia_check.json `floor_seed*` |
| oracle {S,T} / {M,S,T}, phantom-augmented ranks | 0.98573 / 0.99443 | 0.98565 / 0.99435 (432010), 0.98569 / 0.99437 (432030) | nadia_check2.json |
| same oracles with plain ranks (no phantom) | — | 0.98286 / 0.99235 | nadia_check.json `oracle_23/123` |
| all weights, ±1 gauge + W_oh row 0 shifted to 0, 3 digits | 400 numbers, 0.98858 | 400 numbers, 0.98852 / 0.98839 | nadia_check2.json `all400@3d` |
| F3 fitted 364 @3d | 0.99234 | 0.99238 / 0.99252; W_hi = ±1, max 3 sig digits | nadia_check.json, nadia_check2.json |
| F1 fitted 340 @3d | 0.98831 | 0.98848 | nadia_check.json |
| row 3: ladder 374 @4d (I round the stored npz myself) | 0.98933 | 0.98935 / 0.98930 | same |
| row 0: mixprec 374 | 0.98835 | 0.98836 / 0.98841 | same |

All match within seed noise (SE of a 1e6 agreement near 0.99 is about 0.0001; seed-to-seed spread is larger
because a few % of inputs are fragile).

## 2. Null and robustness tests (my own code)

- **Bar seed spread.** Two input seeds, 20 noise draws each: 0.98686 and 0.98697. Per-draw sd 0.0036-0.0046,
  so the 20-draw mean has SE about 0.001. The 3-digit and T20 rows (0.9883-0.9886) clear it by about 1.5 SE.
  That supports "at the bar, inside its uncertainty" for those rows.
- **Is nearest rounding lucky?** All 416 gauge weights (no row shift) at 3 digits: nearest rounding 0.98977;
  5 draws of stochastic rounding at the same 3 digits: 0.9695-0.9885 (nadia_check.json
  `all@3d_stochastic_5draws`). So the 3-digit passes depend on the exact rounding and on the gauge form. The
  W_oh row shift alone moves it from 0.98977 (no shift) to 0.98852 (row 0 shift). This agrees with Hugo's and
  Rasmus's finding that 7 of 10 row shifts fall under the bar. The 4-digit and fitted clear passes do not
  depend on this.
- **New compact family against the "readable rules stop near 96%" claim** (nadia_scorer.py). One shared
  per-position scorer, score_k = MLP(x_k, M, S, T of (0, x), is-max/2nd/3rd flags, one-hot position k,
  one-hot position of the max), trained by cross-entropy on the model's argmax with fresh N(0,1) batches
  (20k steps), scored on seed 432060 (1e6). C1 on the same seed: 0.95522.

  | hidden | numbers (all fitted) | agreement |
  |---|---|---|
  | 3 | 88 | 0.95566 |
  | 6 | 175 | 0.95695 |
  | 24 | 697 | 0.96126 |

  A 175-number fitted scorer reaches 0.957, about the same as C1+ (0.956, 12 literals); 697 numbers give
  0.961. This family was not in RESULTS.md and it lands where the post says compact rules land. It is a
  dev number, not a holdout, and it counts as fitted. It does not change the "best at most 200 numbers"
  line in a way a reader would notice (95.6% vs 95.7%).

## 3. Definitions and claims

1. **Post Limits sentence, 0.3% noise (fixed).** The post said "at 0.3% weight noise the network matches its
   noisy copies only 95.2% of the time, like the 4-number rule". Mine: 0.9611 (10 draws, 1e6), 0.9606 and
   0.9596 (40 draws each, 2e5), per-draw sd 0.016. Hugo (200 draws) 0.9588, Vera re-run (60 draws) 0.9591.
   The 0.9523 was a 10-draw estimate. Correct text: "about 96%, only half a point above the 4-number rule".
   At 0.3% the readable rules still do not pass.
2. **"Ceiling" wording for cell_ceiling.py (internal docs only, not in the post).** research.md:139 says the
   per-cell majority "upper-bounds any rule that reads only those features", and EXPLANATION.md:36-37 says
   "best possible rule ... 0.965 on held-out samples". That is too strong. A held-out majority vote over
   398k cells from 5e6 training samples (about 12 per cell) estimates the best cell rule from below (train
   agreement 0.972 is the optimistic side), and a rule that reads the same features as continuous values
   can beat any binning. Read 0.965 as "what a cell lookup reached", not as a bound. Non-blocking: the post
   does not cite it, and my scorer (0.957-0.961) stays below it.
3. **residual_percfg 0.97062 / post "rules with 63,270 fitted constants reach 97.1%".** "Reach" is the right
   word: it is a least-squares fit on near-ties only, so it is a lower bound on that family, not a ceiling.
   Non-blocking.
4. **Bar change from 0.99 to 0.98671.** Esben chose it in the room ("Reducer til under støjgulvet", #1552),
   and the 0.1% level was fixed before C1 was scored on the holdout. Not a stretched definition.

## 4. Novelty

Searched the web (ARC post, LessWrong/AF comments, GitHub gists, recent results). Prior work on this model:
ARC's post, Zihao Chen's notes, Goldowsky-Dill and Keane comments and gists, all cited in the post. I found
no published short description that matches the model's own errors, and no claim of agreement with the model
above 0.96 for a short rule. The post claims no "first" and states that it does not address ARC's
mechanistic-estimate challenge. Novelty is not overstated.

## 5. What I did not check

- The bits accounting (Hugo did it independently).
- Holdout numbers (Rasmus only).
- The mechanism claims in EXPLANATION.md s.2-4 (neuron roles, clock period 8.4).
