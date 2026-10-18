"""Vera: independent recompute of the T2 probe gate from Bruno's saved clean activations (no Bruno code used).
Run from the work dir: uv run --with numpy --with scikit-learn python swarm-theory-1018-shift-spread/skeptic/vera_gate.py
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = "kran-research/projects/rethreshold-probe-evasion/data/"
z = np.load(D + "clean_acts.npz")
X, y, split, score = z["X"].astype(np.float64), z["y"], z["split"], z["score"]
p = np.load(D + "probe.npz")
w, b = p["w"], float(p["b"])
rows = json.load(open(D + "split.json"))

# 1. labels and splits agree with split.json; positive = FALSE statement
assert all(r["split"] == s for r, s in zip(rows, split))
assert all(r["pos"] == yy == 1 - r["true"] for r, yy in zip(rows, y))
# 2. no city in two splits
cs = defaultdict(set)
for r in rows:
    cs[r["city"]].add(r["split"])
print("cities", len(cs), "in >1 split:", sum(len(v) > 1 for v in cs.values()))
# 3. score = w.X + b
s = X @ w + b
print("max |score - (Xw+b)|", np.abs(s - score).max())


def thr(neg, k=2):
    return np.sort(neg)[-k]


rn = s[(split == "refit") & (y == 0)]
en, ep = s[(split == "eval") & (y == 0)], s[(split == "eval") & (y == 1)]
t0 = thr(rn)
print("n refit_neg %d eval %d/%d" % (len(rn), len(en), len(ep)))
print("t0 %.6f  TPR_eval %.4f  FPR_eval %.4f  AUROC_eval %.5f" % (
    t0, (ep > t0).mean(), (en > t0).mean(), roc_auc_score(np.r_[np.zeros(len(en)), np.ones(len(ep))], np.r_[en, ep])))
print("t0 (5%%, 10th) %.4f TPR %.4f" % (thr(rn, 10), (ep > thr(rn, 10)).mean()))

# 4. my own probe on the probe split only (standardised, C grid), compare direction
m = split == "probe"
mu, sd = X[m].mean(0), X[m].std(0) + 1e-8
for C in (0.01, 1.0):
    lr = LogisticRegression(C=C, max_iter=5000).fit((X[m] - mu) / sd, y[m])
    wv = lr.coef_[0] / sd
    sv = X @ wv
    t = thr(sv[(split == "refit") & (y == 0)])
    print("my probe C=%-5s cos(w,Bruno w)=%.4f  eval TPR@t0 %.4f AUROC %.5f" % (
        C, wv @ w / np.linalg.norm(wv) / np.linalg.norm(w),
        (sv[(split == "eval") & (y == 1)] > t).mean(),
        roc_auc_score(y[split == "eval"], sv[split == "eval"])))
