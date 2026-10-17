# T17 — quantization and description bits for the network-shaped descriptions (Felix, round 2)

All numbers on dev seed 61 (2e5), agreement with the model's argmax. Holdout only via Rasmus.
Bar = noise floor 0.98671. These are fitted or pruned networks, not hand algorithms (PREREG item 4: all
have > 200 literals and the full 16-neuron recurrence).

## Result

- **Quantization-aware finetune does not help.** Straight-through (STE) rounding to a fixed-point grid,
  KL to the model's logits, best checkpoint kept: 340-entry net (Bruno's prune_ft_T1_200_140) at
  10 bits/entry 0.97949 → 0.97953; at 12 bits 0.98705 → 0.98705 (no change). Without the best-checkpoint
  guard, STE made it worse (8 bits 0.872 → 0.846, 10 bits 0.972 → 0.963 on the T5 net). Bruno found the
  same for 3 significant digits (STE 0.982 → 0.972–0.980; progressive freezing 0.98500).
- **Fixed-point grids must live in the rms-1 gauge.** In the W_hi = ±1 gauge column scales differ by
  orders of magnitude; a per-row grid at 6 bits zeroes half the entries (agreement 0.09).
- **Smallest description at the floor we found: about 5.3 kbit.** Code length (sign + mantissa +
  exponent entropy per free number, plus an enumerative sparsity mask; `bits_curve.py`):

| net | free numbers | digits | bits | dev-61 agreement | holdout (Rasmus) |
|---|---|---|---|---|---|
| ladder_pruned_0.995 (original weights, pruned) | 382 | 3 | **5,293** | 0.98743 | 0.98673 (tie) |
| original 432 weights, ±1 gauge | 400 | 3 | 5,479 | 0.98897 | not run (round 1 rounded the raw, ungauged weights: 0.982–0.986) |
| prune_ft_T1_200_140 fixed-point 12 bit (`_fp12.npz`) | 340 + 41 scales | — | 5,632 | 0.98705 | not run |
| prune_ft_T1_200_140 (fitted) | 340 | 4 | 5,946 | 0.98772 | not run |
| ladder_pruned_0.99 | 374 | 4 | 6,458 | 0.98945 | 0.98933 |
| prune_ft_T5_200_140 (fitted) | 340 | 3 | 4,819 | 0.98184 | — (below floor) |
| prune_ft_T1_200_140 (fitted) | 340 | 3 | 4,817 | 0.97379 | — (below floor) |
| any net above | 290–400 | 2 | 3,200–4,100 | 0.64–0.74 | — |

Reference: C1 (`algorithm_compact.py`) is a few lines with 4 literals and gets 0.955.
The jump from 2 to 3 digits (0.70 → 0.98) at almost every size is the same fact as round 1: the
model's decisions hang on the 3rd significant digit of its weights.

## Files

- `prune_qatF.py` — fixed-point STE finetune (rms-gauge grid, per-row steps, best checkpoint on dev 61).
  Outputs `prune_ft_T5_200_140_fp{8,10}.*` (first runs, no checkpoint guard) and
  `prune_ft_T1_200_140_fp{10,12}.*` (lr 3e-5, guard).
- `bits_curve.py` → `bits_curve.json` (7 s).
- `results_ledger.py` → `RESULTS_ledger.json` (all proposals, one row each).
