"""Vera rebuild of Iris' T22 'C1+' from its stated clauses only (#1999), not from her code.
C1 = phantom-augmented 2nd argmax (phantom -> position 0). Clauses, in order of precedence (last wins):
 (1) 3rd (augmented) at position 9 and S - T < 0.01 |M| -> pick the 3rd
 (2) exactly one positive input, at position 2 -> pick 1
 (3) no positive input -> pick 0
Dev seeds 950, 951 (1e6 each).
"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_core import load, pred, sample, aug_order, HERE

W = load(); out = {}
for seed in (950, 951):
    X = sample(seed, 1_000_000); y = pred(X, W)
    pos, ph = aug_order(X)
    A = -np.sort(-np.concatenate([np.zeros((len(X), 1)), X], 1), 1)
    c1 = pos[:, 1].copy(); p = c1.copy()
    third_real = ~ph[:, 2]
    m1 = third_real & (pos[:, 2] == 9) & ((A[:, 1] - A[:, 2]) < 0.01 * np.abs(A[:, 0]))
    p[m1] = 9
    npos = (X > 0).sum(1)
    m2 = (npos == 1) & (X[:, 2] > 0); p[m2] = 1
    p[npos == 0] = 0
    true2 = np.argsort(-X, 1)[:, 1]; errset = y != true2
    out[seed] = {"C1": float((c1 == y).mean()), "C1plus": float((p == y).mean()),
                 "gain_pt": float(100 * ((p == y).mean() - (c1 == y).mean())),
                 "errset_C1": float((c1 == y)[errset].mean()), "errset_C1plus": float((p == y)[errset].mean()),
                 "fires": [int(m1.sum()), int(m2.sum()), int((npos == 0).sum())]}
json.dump(out, open(f"{HERE}/skeptic/vera_c1plus.json", "w"), indent=1)
print(json.dumps(out, indent=1))
