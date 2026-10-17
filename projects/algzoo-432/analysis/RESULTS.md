# RESULTS — ARC AlgZoo M16,10 (432 parameters, 2nd argmax), rounds 1 and 2

Final, 2026-10-17 23:30 Copenhagen time. Drafted by Felix (T19) from `RESULTS_ledger.json` (58 rows), checked by
Magnus; every holdout number was run once by Rasmus; skeptic rebuild by Vera (`skeptic/signoff.md`).

## Is it solved?

**No, not in the strict sense.** No short description (at most 200 numbers, no copy of the 16×16
recurrence) matches the model on at least 0.98671 of inputs. That bar is the model's own noise floor: the
mean agreement of the model with a copy of itself with 0.1% weight noise.

The best answers we have:

1. **Shortest description that reaches the bar: a rounded copy of the network itself, about 5–6.5 kbit.**
   The bar itself is an estimate: Rasmus measured 0.98671 (holdout, 20 noise draws). With 100–200 draws on
   other seeds: Hugo 0.98717 (SE 0.00026), Vera 0.98770 and 0.98666 (SE ~0.0003); Vera's earlier 20-draw
   0.98855 was pulled high by the skewed draw distribution. So the bar estimates span 0.9867–0.9877. Against that band:
   - **Fewest bits with a clear pass:** a retrained (fitted) net with 364 numbers at 3 digits, 5,084 bits,
     holdout 0.99234. All 364 numbers are fit to the model's outputs (distillation, not explanation).
   - **Zero-fitted clear pass:** 374 pruned original weights at 4 significant digits, 6,458 bits,
     holdout 0.98933 (382 at 4 digits: 6,562 bits, 0.99446).
   - **Pass narrowly (0.0006–0.0019 above every bar estimate):** fitted 340 numbers at 3 digits, 4,814
     bits, 0.98831; 374 original weights with per-entry digits (T20), 5,208 bits, 0.98835; all 400
     original weights at 3 digits, 5,479 bits, 0.98858 (depends on the W_oh row-shift form, see A).
   - **Tie:** 382 at 3 digits, 5,293 bits, 0.98673.
   - Nothing at or under 200 numbers comes close (best 0.957 dev: Nadia's fitted per-position MLP scorer,
     175 numbers; C1+ 0.956 holdout).
2. **Best readable algorithm:** C1 = "the 2nd largest of (0, x_0, …, x_9), with the leading 0 mapped to
   position 0". 4 literals, 0 fitted constants, holdout 0.95503. With 3 extra clauses (C1+, 5 fitted
   numbers): holdout 0.95634.
3. **Best rule with fitted constants:** a linear comparator of the max, 2nd and 3rd candidates per
   position configuration, 63,270 fitted constants: 0.97062 on a dev seed. Still 1.6 points under the bar.

The round-2 pre-registration (`PREREG_round2.md`, fixed 22:53 before any round-2 result) predicted "no;
the best short description stays below 0.975", at 70% confidence. The prediction held: the best
descriptions with at most 200 numbers reach 0.956-0.957 (C1+ 0.95634 holdout; Nadia's fitted 175-number
per-position scorer 0.9570 dev, 88 numbers 0.9557, 697 numbers 0.9613; `skeptics/nadia/`). The 0.97062 rule
has 63,270 constants, so it is not short and does not test the prediction.

All agreement numbers are with the model's argmax, on N(0,1) inputs. Holdout = seed 432432, N = 1e6,
`check_agreement.py`, run by Rasmus only. Dev numbers come from other seeds and are marked "dev".
"Robust" and "fragile" come from Iris's T18 (`robust_eval.md`, dev seed 83, 1e6): an input is fragile
when the model's answer changes under at least one of 20 draws of the 0.1% bar noise (5.87% of inputs).
On robust inputs, a noisy copy of the model agrees 0.99905; on fragile inputs 0.7683.

## A. Candidates that reach the bar, ranked by fewest bits

Bar estimates span 0.9867–0.9877 (100–200 draws; see top). Clear passes (≥ 0.9890): F3, rows 3 and 4.
Narrow passes, above every bar estimate by 0.0006–0.0019: F1, row 0, row 2 (rows 0 and 2 carry the row-shift
caveat below). Ties inside the band: row 1 (0.98673), F4 (0.98718), F5 (0.98777). F2 has no holdout run.
Rows F1–F5 are retrained (fitted) networks: fewer bits, but every number
is fit to the model's outputs. Rows 0–4 are the original weights, pruned and rounded (zero fitted constants).
Bits = 1 sign bit + mantissa bits + exponent entropy per free number, plus a sparsity mask
(`bits_curve.py`, `bits_curve.md`). These are networks, not hand algorithms: all keep the 16-neuron
recurrence and have more than 200 numbers.

| # | candidate | owner | numbers | fitted | bits | holdout | dev | robust / fragile (dev 83, 1e6) | file |
|---|---|---|---|---|---|---|---|---|---|
| F1 | prune + long finetune, 340 numbers, 3 digits (freeze-rounding) | Bruno | 340 | 340 | 4,814 | **0.98831** | 0.98827 (dev 62) | 0.99849 / 0.8242 | prune_ft_long_200_140_q3f.npz |
| F2 | same, 350 numbers, 3 digits | Bruno | 350 | 350 | 4,941 | not run | 0.98862 (dev 62) | 0.99875 / 0.8290 | prune_ft_long_* |
| F3 | same, 364 numbers, 3 digits | Bruno | 364 | 364 | 5,084 | **0.99234** | 0.99250 (dev 62) | 0.99970 / 0.8769 | prune_ft_long_220_144_q3f.npz |
| 0 | original weights, per-entry digits 0–4 (T20) | Felix | 374 | 0 | **5,208** | **0.98835** | 0.98921 (sel.) / 0.98827 (dev 62) | — | mixprec.py, mixprec_0.989.npz |
| 1 | original weights, greedy-pruned, 3 digits | Felix | 382 | 0 | 5,293 | **0.98673** (tie) | 0.98743 | 0.99851 / 0.7979 | ladder.py, ladder_pruned_0.995.npz |
| 2 | all original weights, ±1 gauge, 3 digits | Bruno, Rasmus, Kasper, Felix | 400 | 0 | 5,479 | **0.98858** | 0.98897 | 0.99946 / 0.8110 | bits_curve.md |
| F4 | prune + finetune net, fixed-point 12 bit | Felix | 340 + 41 scales | 381 | 5,632 | 0.98718 | 0.98705 | — | prune_ft_T1_200_140_fp12.npz |
| F5 | prune + finetune net, 4 digits | Bruno | 340 | 340 | 5,946 | 0.98777 | 0.98772 | 0.99847 / 0.8183 (exact) | prune_ft_T1_200_140.npz |
| 3 | original weights, greedy-pruned, 4 digits | Felix | 374 | 0 | 6,458 | **0.98933** | 0.98945 | 0.99757 / 0.8601 | ladder_pruned_0.99.npz |
| 4 | original weights, greedy-pruned, 4 digits | Felix | 382 | 0 | 6,562 | 0.99446 | 0.99470 | 0.99892 / 0.9270 | ladder_pruned_0.995.npz |

Fitted = the numbers were trained on the model's outputs (distillation); they count as fitted, not
explained (PREREG item 3).

Row 1 sits on the bar on every seed we tried (holdout 0.98673, dev 61 0.98743, dev 83 0.98674).
Near misses below the bar: Bruno's fitted nets at 325 / 310 / 295 numbers, 3 digits, 4,621–4,266 bits:
0.98301 / 0.97886 / 0.96563 (dev 62); 374 numbers at 3 digits 0.98407 (holdout); the 340-number fitted net at 3 digits 0.98500 with progressive freezing (Bruno) and
0.98184 without; at 2 digits every network is at 0.64–0.74.

Convention caveat (Hugo, confirmed by Rasmus #2142): the 3-digit rows of the original weights (rows 0–2)
depend on which of the 10 equivalent W_oh row shifts gets rounded. `bits_curve.py` fixes shift 0 by
convention, and shift 0 happens to be the best: over all 10 shifts, all-400 at 3 digits gives
0.98381–0.98873 (dev 4321), and 7 of 10 fall below the bar. The 4-digit rows (3, 4) and the fitted rows
clear the bar by more than 0.002 and do not depend on this.

Noise-floor checks: Hugo's 200-draw 0.1% floor 0.98717 (SE 0.00026, dev 7000-7140) supports the bar
0.98671. At 0.3% noise the floor is about 0.960 (Nadia, 80 draws; Vera's 10-draw 0.952 was low), half a
point above C1.

## B. Readable and compact candidates, ranked by agreement

Where the readable rule is faithful (Iris, `certified_domain.py`, dev 82, 1e6): on inputs whose top-3 gaps
M−S and S−T both exceed 5% of rms(x) (82.2% of inputs), C1 agrees 0.98734 with the model; outside that
domain it agrees 0.806. At 10% of rms: 67.1% of inputs, 0.99308. This does not put C1 at the floor inside
the domain: a 0.1%-noise copy of the model agrees 0.9962 there (Vera, `skeptic/vera_domain.py`, dev 930),
so C1 misses 3.4 times as often. The domain bound is post hoc (tau not frozen in advance).


| # | candidate | owner | literals | fitted | holdout | dev | robust / fragile (dev 83, 1e6) | file |
|---|---|---|---|---|---|---|---|---|
| 0 | C1+ = C1 + 3 clauses (3rd at pos 9 on near tie; one positive at pos 2 → 1; no positive → 0) | Iris | 12 | 5 (fitted: threshold 0.01, positions) | **0.95634** | 0.95682 (dev 84) | 0.97874 / 0.6051 | algorithm_c1plus.py |
| 1 | C1: 2nd argmax with a phantom 0 before x_0 | Felix | 4 | 0 | **0.95503** | 0.95582 | 0.97789 / 0.5963 | algorithm_compact.py |
| 2 | null: true 2nd argmax | — | 0 | 0 | 0.95283 | 0.95349 | 0.97648 / 0.5818 | check_agreement.py |
| 3 | ARC's handcrafted 22-neuron model | ARC | — | 0 | — | 0.9534 | — | research.md |

C1+ adds +0.14 pt; no single clause gains more than 0.1 pt, so the readable family stops near 0.957
(`c1plus_search.json`). C1 matches the model where the model is right (0.998) and on the phantom-zero errors, but only about 8.5% of the
model's other errors (0.084–0.086 on dev and holdout). On robust inputs it misses 2.2 points; those misses are systematic near-tie reversals
(the model picks the 3rd in 74.5%, the max in 17.3%; `residual_search.md` s.2).

## C. Rules with many fitted constants (discovery, not candidates)

All fit on dev seeds; none is short.

| family | owner | fitted constants | dev | robust / fragile | file |
|---|---|---|---|---|---|
| candidate MLP 3×256, still improving | Iris | 143,873 | 0.97400 | 0.99213 / 0.6925 | residual_mlp.py |
| per-config linear over M/S/T, full feature bank | Iris | 63,270 | 0.97062 | 0.98890 / 0.6820 | residual_percfg.py |
| per-config linear S-vs-T on x (+ relu, running maxes) | Iris | 9,000 (33,300) | 0.96569 (0.96825) | 0.98542 / 0.6556 | residual_percfg.py |
| per-cell majority on top-3 positions + binned gaps + signs | Iris | 398,000 cells | 0.9651 | — | cell_ceiling.py |
| ReLU pattern from 10 streaming values (160 trees) + exact weights | Felix | trees | 0.9626 | — | pattern_linear.md |
| linear logits per (pM, pS, pT) class | Felix | 87,100 | 0.9592 | — | pattern_linear.md |
| gradient-boosted trees on rank/gap/value features | Bruno, Kasper | thousands | 0.934–0.960 | — | scratch/gbm_ceiling.py |
| 2nd argmax of G·x (10 gains / full 10×10) | Bruno | 10 / 100 | 0.9543 / 0.9517 | — | scratch/distorted_2nd_argmax.py |
| ReLU pattern from prefix order (160 trees) + exact weights | Felix | trees | 0.938 | — | pattern_linear.md |

## D. Oracle ceilings (what a perfect chooser could get)

| oracle | dev | source |
|---|---|---|
| perfect choice inside {2nd, 3rd} | 0.98573, under the bar | residual_search.md s.1 |
| perfect choice inside {2nd, 3rd, 4th} | 0.98821 | same |
| perfect choice inside {max, 2nd, 3rd} | 0.99443 | same |
| true ReLU pattern + per-pattern linear map | 1.0000 (by construction) | pattern_linear.md |

So any "near-tie corrector" that only swaps the 2nd and 3rd cannot pass. A passing rule must also predict
when the model picks the max.

## E. Negative tracks and what each one showed

- **Pattern-predicted linear model (Felix, T15).** The ReLU pattern is not a function of the rank order
  (0.938 even with the exact weights downstream). The oscillator neurons (n3, n10–n15) are the hard part.
- **Correction terms on top of C1 (Iris, T16).** No 1–5 term fix exists in the families tried; the
  comparator changes form inside one position configuration.
- **Prune + finetune below 340 numbers (Bruno, N1).** 290 numbers 0.962, 250 0.904–0.911, 200 0.72–0.83
  (slow ramp 0.830), 180 0.48. Removing neurons: 15 → 0.960, 12 → 0.549, 8 → 0.326. Below about 250
  numbers the nets fall under the zero-parameter true 2nd argmax (0.953), so the low end is limited by the
  optimizer and is not a size bound.
- **Quantization-aware training (Felix T17, Bruno N1b).** Straight-through rounding gives about zero gain
  (12 bit 0.98705 → 0.98705; 3 digits 0.982 → 0.972–0.980). Progressive freezing gives 0.98500.
- **Distillation into small MLPs and RNNs (Bruno, T14).** Uninformative: a same-size student reached
  only 0.702.
- **Linearizing neurons, keeping top-k inputs, integer gains (round 1).** All below 0.98; see
  EXPLANATION.md s.3.

## F. Mechanism (what we do understand)

Short version; details and tests in `EXPLANATION.md` s.2–4.

- Subcircuit A {1, 2, 4, 6, 7, 8} keeps a leaky running max (n2, leak 0.969), a new-max increment (n4),
  "max minus last input" (n7), and a shift register of deficits "the max was 1 / 2 / ≥3 inputs ago"
  (n6, n1, n8).
- Gates: n9 = "last input was a new max" (swap 0.9953), n0 = "max not exactly 2 steps ago" (0.9968),
  n5 = comparator against about 0.8 × the earlier 2nd max (0.9917, fitted).
- B {3, 10–15} is an oscillator bank with a "max clock" (period 8.4 steps) whose phase tracks steps since
  the max, plus weaker "2nd clocks".
- The leading phantom 0 (h_0 = 0, no biases) explains about 0.8 points of the model's errors exactly.
- The model's remaining choices on near ties depend on the 3rd significant digit of its weights: 3 digits
  gives 0.98–0.99, 2 digits gives 0.64–0.74.

## G. What is missing

- No proof that a short rule cannot exist. The evidence covers rank/gap/position rules, linear comparators
  per configuration, trees and pattern rules. A rule built on some feature we did not try is still open.
- This is not ARC's open challenge (a mechanistic estimate of accuracy that beats sampling). We do not
  claim to solve it.
- The model was trained on Unif[-1, 1]; all numbers here use N(0, 1) inputs, as in the ARC post.
