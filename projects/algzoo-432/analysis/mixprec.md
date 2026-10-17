# T20 — per-entry mixed precision of the original weights (Felix, round 2)

`mixprec.py`: every one of the 400 free numbers of the ORIGINAL weights (±1 gauge) gets the fewest
significant digits d ∈ {0 = pruned, 1, 2, 3} that keeps agreement on the selection set (dev seed 61, 1e5)
≥ target, else d = 4; entries visited from least to most sensitive. Zero fitted constants (every kept
number is an original weight, rounded). Checked on dev seed 62 (2e5). Holdout only by Rasmus.

| target (dev 61) | kept | digits 0/1/2/3/4 | bits | dev 61 | dev 62 | file |
|---|---|---|---|---|---|---|
| 0.989 | 374 | 26 / 74 / 115 / 134 / 51 | **5,208** | 0.98921 | 0.98827 | mixprec_0.989.npz (holdout 0.98835, Rasmus #1982) |
| 0.9885 | 373 | 27 / 67 / 122 / 127 / 57 | 5,243 | 0.98852 | 0.98812 | mixprec_0.9885.npz |

Reference (bits_curve.md): all 400 at 3 digits 5,479 bits (holdout 0.98858); 382 pruned at 3 digits
5,293 bits (holdout 0.98673).

Reading: per-entry precision saves only ~5% of the code (5,479 → 5,208 bits at the same agreement).
74 entries survive with 1 digit and 26 can be dropped, but 185 entries still need 3–4 digits, and the
per-entry digit level itself costs ~2.1 bits × 400 slots. The description at the bar stays about 5 kbit.
Code length rule as in bits_curve.md, with the digit-level entropy replacing the sparsity mask.

Holdout (Rasmus #1982): 0.98835. That passes the fixed bar 0.98671 but sits inside the floor's own
uncertainty (Vera's floor estimates 0.98813–0.98855, #1993). Only 374 numbers at 4 digits (0.98933,
6,458 bits) clears every floor estimate.
