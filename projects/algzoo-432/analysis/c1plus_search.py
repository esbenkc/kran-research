"""T22 (Iris): greedy search for single slice clauses on top of C1 (dev seed 81 fit, dev seed 82 check).

Clause = 'if the 3rd (or the max) is at position j [after/before S] and its gap to S is < delta*|M|, pick it'.
Gain = change in agreement with the model if the clause is added alone to C1. Also the two slice rules for
inputs with 0 or 1 positive value. Usage: .venv/bin/python c1plus_search.py   (~20 s) -> c1plus_search.json
"""
import json
import numpy as np
from residual_common import data, aug_order

DELTAS = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]


def stats(seed):
    X, L, m = data(seed, 1_000_000)
    o = aug_order(X); O = np.maximum(o, 0)
    v = np.take_along_axis(np.c_[np.zeros(len(X)), X], o + 1, 1)
    Ma = np.abs(v[:, 0]) + 1e-12
    return dict(X=X, m=m, o=o, O=O, g12=(v[:, 0] - v[:, 1]) / Ma, g23=(v[:, 1] - v[:, 2]) / Ma, npos=(X > 0).sum(1))


def gain(St, tgt, j, delta, order):
    k = {"T": 2, "M": 0}[tgt]; g = St["g23"] if tgt == "T" else St["g12"]; o, O, m = St["o"], St["O"], St["m"]
    sel = (g < delta) & (O[:, k] != O[:, 1])
    if j is not None: sel &= o[:, k] == j
    if order == "after": sel &= o[:, k] > o[:, 1]
    if order == "before": sel &= o[:, k] < o[:, 1]
    return float(((m[sel] == O[sel, k]).sum() - (m[sel] == O[sel, 1]).sum()) / len(m))


A, B = stats(81), stats(82)
rows = []
for tgt in "TM":
    for j in [None] + list(range(-1, 10)):
        for order in (None, "after", "before"):
            g, d = max((gain(A, tgt, j, d, order), d) for d in DELTAS)
            if g > 0:
                rows.append(dict(pick=tgt, pos=j, order=order, delta=d, gain81=round(g, 5), gain82=round(gain(B, tgt, j, d, order), 5)))
rows.sort(key=lambda r: -r["gain81"])
slices = []
for St, seed in ((A, 81), (B, 82)):
    X, m, npos = St["X"], St["m"], St["npos"]
    pm = X.argmax(1)
    for r, pk in ((2, 1), (3, 2), (7, 7)):
        s = (npos == 1) & (pm == r)
        slices.append(dict(seed=seed, rule=f"one positive at {r} -> pick {pk}", gain=round(float(((m[s] == pk).sum() - (m[s] == 0).sum()) / len(m)), 5)))
    z = npos == 0
    slices.append(dict(seed=seed, rule="no positive -> pick 0", gain=round(float(((m[z] == 0).sum() - (m[z] == X[z].argmax(1)).sum()) / len(m)), 5)))
for r in rows[:12]: print(r)
for s in slices: print(s)
json.dump(dict(clauses=rows, slices=slices), open("c1plus_search.json", "w"), indent=1)
