# Vera (skeptic) — round-2 sign-off, algzoo-432

Claim checked (Magnus #1933): "No description of ARC's 432-parameter 2nd-argmax RNN with at most 200 numbers
reaches the noise-floor bar 0.98671. Readable rules cap near 0.955-0.971. Network-shaped descriptions need
about 340-382 numbers."

Verdict: **CONFIRMED as "none found"**, with four wording conditions for the Lab post (section 4).
It is not evidence that no such description exists (the searches are optimizer-limited below ~290 numbers).

All numbers below are mine: own forward from `m16_10.pth` (`vera_core.py`), dev seeds 900-921, never the
holdout 432432. Scripts live in `swarm-algzoo-432/skeptic/` (mirror: `projects/algzoo-432/analysis/skeptic/`).
Run from that folder's parent with `.venv/bin/python skeptic/<script>.py`.

## 1. Rebuilt numbers

| quantity | team | mine | source |
|---|---|---|---|
| forward vs torch.nn.RNN(relu, no bias) | — | max diff 0.0 | vera_checks.py:18-24 |
| model accuracy = null (true 2nd argmax) agreement | 0.95283 (holdout) | 0.95299 (seed 901, 1e6) | vera_checks.py:30 |
| phantom-zero 2nd argmax (C1 logic, my own code) | 0.95503 (holdout) | 0.95504 (seed 901) | vera_checks.py:31, vera_core.aug_order |
| oracle {S,T} | 0.98573 | 0.98557 | vera_checks.py:37-44 |
| oracle {M,S,T} | 0.99443 | 0.99438 | same |
| oracle {S,T} without the phantom | — | 0.98278 | vera_checks.py:46-52 |
| noise floor, 0.1%, 20 draws | 0.98671, sd 0.0042 (holdout) | 0.98855, sd 0.0025 (seed 902, 1e6) | vera_checks.py:55-62 |
| noise floor, 0.1%, 10 draws | — | 0.98813, sd 0.0037 (seed 910, 4e5) | vera_checks2.py |
| noise floor, 0.03% | 0.99610 | 0.99622 / 0.99564 | both |
| fragile share (20 draws) | 5.87-5.97% | 5.72% | vera_checks2.py |
| C1 logic on robust / fragile inputs | 0.9779 / 0.587 | 0.9776 / 0.587 | vera_checks2.py |
| noisy copy on robust / fragile | 0.99905 / ~0.78 | 0.99957 / 0.798 | vera_checks2.py |
| Bruno 340-literal fitted net (T1_200_140), exact / 3 digits | 0.98772 / 0.97379 | 0.98779 / 0.97396 (seed 904) | vera_checks.py:65-81 |
| Bruno T5_200_140_q3f (340 x 3 digits) | 0.98500 | 0.98511 | same |
| Bruno slow ramp, 200 literals | 0.830 | 0.8284 | same |
| Bruno 290 literals (T5_160_130) | 0.962 | 0.9615 | same |
| ladder_pruned_0.995: nonzeros | 382 | 16 W_hi (±1) + 245 W_hh + 137 W_oh = 382 | same |
| ladder_pruned_0.995 exact / 3 digits | — / 0.98743 (dev 61) | 0.99509 / 0.98650 (seed 904) | same |
| ladder_pruned_0.99 exact | 0.98976 | 0.98979 (374 nonzero non-±1) | same |

Every number the team posted that I rebuilt reproduces within sampling noise.
Holdout leak check: `grep 432432` over all round-2 `.py` files and `scratch/` finds the seed only in
comments, in `assert seed != 432432` (residual_common.py:25) and in `check_agreement.py`. No leak found.

## 2. New attack: low-rank family (nobody tried it)

`vera_lowrank.py`: W_hh = U V^T (rank r), W_oh = B C^T (rank k), W_hi gauged to ±1, SVD warm start in the
rms gauge, KL finetune to the model's logits, 3000 steps x 4096 fresh inputs, eval seed 921 (2e5).

| r, k | free numbers | SVD init | after finetune |
|---|---|---|---|
| 4, 4 | 200 | 0.102 | 0.171 |
| 5, 2 | 183 | 0.107 | 0.154 |
| 3, 5 | 192 | 0.102 | 0.165 |
| 14, 9 (control) | 405 | 0.939 | 0.991 |

The spectrum of W_hh is flat (singular values 5.8, 4.4, 2.9, 2.5, 2.1, 1.7, ... 0.32, 0.20, then 0.024,
0.001), so the network is close to rank 14 but not to rank 4-5. Low rank is worse than pruning at 200
numbers. It does not change the picture. (Short training; weak evidence on its own.)

## 3. How fair is the bar?

1. **The bar is steep in the noise level.** Agreement of the model with a noisy copy (10 draws, seed 910):
   0.03% 0.9956, 0.1% 0.9881, 0.2% 0.9712, 0.3% 0.9523, 0.5% 0.9384, 1% 0.805.
   **Correction (23:41, after Nadia #2137):** 10 draws were too few at 0.3% (per-draw sd 0.013).
   60 draws, seed 970 (2e5), noise rng 971 (`vera_noise60.json`): 0.2% 0.9745 (SE 0.0011),
   **0.3% 0.9591 (SE 0.0016)**. Nadia's 80-draw value 0.9596-0.9606 agrees. So at 0.3% a noisy copy is
   still ~0.4 pt above the 4-number rule (0.955); the rule's equivalent noise level is ~0.35-0.4%.
   Pass/fail still depends on the 0.1% choice. The 0.1% level was fixed before C1 existed
   (PREREG.md amendment), so this is not post-hoc, but the post must say it.
2. **The bar itself has error bars of about ±0.002.** Three independent 0.1% estimates: 0.98671
   (holdout), 0.98855 (seed 902), 0.98813 (seed 910). Draw-to-draw sd is 0.0025-0.0042, so the 20-draw
   mean has SE ~0.0006-0.0009, and the spread between estimates is 0.0018. Candidates at 0.9867-0.9886
   (382@3 digits 0.98673, fp12 0.98718, 340 fitted 0.98777, all-400@3 digits 0.98858) are ties with
   the bar, not passes. Only ladder 374@4 digits (0.98933) is clearly above it.
3. **"The deciding detail sits in the 3rd digit" is close to circular.** 0.1% multiplicative noise is
   by construction a 3rd-significant-digit perturbation (3-digit rounding error is 0.05-0.5% relative).
   A bar defined at that precision will say 2 digits fail and 3 digits tie. The independent fact is
   the robust/fragile split: on the 94% of inputs that survive all 20 noise draws, C1 still misses 2.2 pt
   while a noisy copy misses 0.04 pt. That part does not depend on the noise level argument.
4. **3-digit rounding depends on the gauge.** Raw ungauged weights at 3 digits: 0.98215 (seed 904),
   below every bar estimate. ±1 gauge at 3 digits: 0.98858 (Rasmus, holdout). 2 digits raw: 0.679.
   "Rounding every weight to 3 digits keeps 98.9%" is true only after the gauge rescaling.
5. **Prereg test uses the wrong row.** PREREG_round2 asks about descriptions with at most 200 numbers.
   The best such description we have is C1 at 0.955 (holdout). The 0.971 rule has 63,270 fitted
   constants and the 0.974 MLP 143,873, so neither is a "short description" or a "readable rule".
   "Readable rules cap ~0.955-0.971" stretches the word readable; the readable cap is 0.955.

## 4. Conditions for the Lab post (draft `_notes/algzoo-432.md` as of 23:16)

- C1 "The prediction held: the best rule we found reaches 97.1%, and it needs 63,270 fitted constants."
  must test the prereg on its own population: best description of at most 200 numbers = 95.5% (C1). The
  63k-constant rule may appear, labelled as a fitted ceiling, not as the prereg outcome.
- C2 "Rounding every weight to 3 significant digits keeps 98.9%" needs "after rescaling each neuron"
  (raw weights give 98.2%).
- C3 "reaches the bar with 340 numbers (98.78%)": say it ties the bar. The bar's own estimate varies by
  ±0.002 across noise draws and input seeds.
- C4 Limits section must state the noise-level dependence: at 0.3% weight noise the network agrees
  with its noisy copy about 96% (corrected from 95.2%, see 3.1), half a point above the 4-number rule.
- Minor: "every network-shaped description misses less than 0.3%" on robust inputs only holds for
  descriptions at the bar (200-literal net: 0.853; 290: 0.981). "Descriptions left of the 200 line stay
  near the plain rule": the 200-number nets sit at 0.17-0.83, far below it. TLDR "every description that
  reaches the bar" -> "every description we found that reaches the bar".
- C5 (added 23:23, Iris T21) "C1 matches the model at noise-floor level on the 82% of inputs without a near
  tie" compares an in-domain number with the global floor. `vera_domain.py` (dev 930, 4e5, 10 draws):
  tau 0.05 coverage 0.822, C1 in-domain 0.9872, noisy copy in-domain **0.9962**; tau 0.1 0.9931 vs 0.9981;
  tau 0.2 0.9969 vs 0.9991. C1 misses 3.4x more than a noisy copy even outside near-ties. The post may
  say "C1 agrees 98.7% outside near ties"; it may not call that noise-floor level. tau 0.05 was also
  picked after seeing results (Iris' own caveat).
