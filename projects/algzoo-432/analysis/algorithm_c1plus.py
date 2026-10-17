"""C1+ (Iris, T22): C1 plus three slice clauses. Contract (check_agreement.py): predict(X float64 [N,10]) -> int [N].

C1 = position of the 2nd largest of (0 at virtual position -1, x_0..x_9); the phantom decodes to position 0
(algorithm_compact.py). The clauses below were FOUND by a greedy search on dev seed 81 over clauses of the
form 'in slice Z pick candidate Y instead of S' and checked on dev seed 82. They are fitted, not derived from
weights. Fitted numbers: one threshold (0.01) and three position choices (9, 2 -> 1, 0).

  1. Late 3rd wins near-ties: if the 3rd largest sits at the last position (9) and S - T < 0.01 |M|,
     pick the 3rd. (In the model, a last input that nearly ties the 2nd usually wins; dev82 gain +0.064 pt.)
  2. One positive input at position 2: the 2nd is the phantom, and the model decodes it as position 1,
     not 0. (dev82 gain +0.053 pt.)
  3. No positive input: the phantom is the max; the model outputs position 0 (51-53% of this 0.1% slice,
     vs 30% for C1's answer). (dev82 gain about +0.02 pt.)
"""
import numpy as np


def predict(X):
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    Xp = np.concatenate([np.zeros((n, 1)), X], 1)
    o = np.argsort(-Xp, 1, kind="stable") - 1  # positions -1..9 by value, phantom at -1
    v = np.take_along_axis(Xp, o + 1, 1)
    pM, pS, pT = o[:, 0], o[:, 1], o[:, 2]
    M, S, T = v[:, 0], v[:, 1], v[:, 2]
    pred = np.maximum(pS, 0)  # C1
    late_tie = (pT == 9) & (S - T < 0.01 * np.abs(M))
    pred = np.where(late_tie, 9, pred)
    npos = (X > 0).sum(1)
    pred = np.where((npos == 1) & (pM == 2), 1, pred)
    pred = np.where(npos == 0, 0, pred)
    return pred
