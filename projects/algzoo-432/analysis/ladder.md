# T12 — minimal-description ladder (owner Felix)

Rebuild: `cd swarm-algzoo-432 && .venv/bin/python ladder.py` (42 s; dev seeds 70 screen 3e4, 71 eval 3e5;
holdout not used). Output `ladder.json`, pruned weights `ladder_pruned_<floor>.npz`.

Question: how few numbers does a simplified copy of the network need to stay at or above the frozen
noise floor (0.98671)? Only exact gauge moves, zeroing and rounding of original weights; nothing refit.
Literal = a nonzero entry that is not exactly ±1.

Exact gauge (Rasmus #1571): per-neuron positive scaling sets all 16 W_hi to ±1; the softmax shift
W_oh <- W_oh - 1·W_oh[0] zeroes row 0. Free numbers left: 432 - 16 - 16 = **400**.

| step | agreement (seed 71) | literals | vs bar |
|---|---|---|---|
| exact gauge | 1.0000 | 400 | ok |
| + round all to 2 significant digits | 0.7069 | 400 | fail |
| + round all to 3 significant digits | 0.9886 | 400 | ok |
| + round all to 4 significant digits | 0.9988 | 400 | ok |
| greedy prune (screen >= 0.997) | 0.9968 | 386 | ok |
| ... + 3 digits | 0.9879 | 386 | ok |
| greedy prune (screen >= 0.995) | 0.9953 | 382 | ok |
| ... + 3 digits | **0.9870** | **382** | ok (smallest passing at 3 digits) |
| greedy prune (screen >= 0.99) | 0.9900 | 374 | ok |
| ... + 3 digits | 0.9843 | 374 | fail |
| ... + 4 digits | 0.9896 | 374 | ok (smallest passing overall) |

Reading:
- The smallest network-shaped description we found at the noise floor (an UPPER bound from greedy
  pruning + uniform rounding, not a proven minimum) is about 374-382 numbers at 3-4
  significant digits (about 10-13 bits each, roughly 4-5 kbit). Greedy zeroing removes only 18-26 of the
  400 free numbers before the floor is hit. PREREG item 4 caps a counted explanation at 200 literals
  and 80 W_hh entries. No step of this ladder comes near that.
- Together with the compact-rule ceilings (C1 0.955; trees 0.9555; top-3-value cell majority 0.965,
  Iris; per-sample exact Jacobian on top-5 values 0.975) this is strong evidence, not proof, that the
  200-literal cap and the 0.98671 bar cannot both be met. The gap needs about 4 kbit of weight detail.

Holdout (Rasmus #1579, seed 432432, N=1e6): 374 numbers at 4 digits 0.98933 (pass); 382 numbers at 3 digits
0.98673 (tie with the floor 0.98671); 386 numbers at 3 digits 0.98771. Mixed precision (Kasper #1588, dev):
A rows 4 digits + rest 2 -> 0.926; A 4 / other 3 / W_oh 2 -> 0.965, so precision is needed in every block.
