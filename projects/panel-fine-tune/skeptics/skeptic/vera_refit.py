"""Vera (1022): own re-fit of country-token probes on ATTACKED activations (PREREG s8, my own code).
For each state and readout r in 10..29: StandardScaler + LogisticRegression(class_weight balanced) on the 1400 training-pool
statements of THAT state; C picked by 5-fold GroupKFold (city) out-of-fold AUROC over {0.01, 0.1, 1, 10} (ties within 0.002 ->
smaller C). Line = 2nd-highest of that state's 200 line-setting true statements; catch on 596 eval false.
Best readout per state = highest out-of-fold AUROC (no eval data). Also: catch at every C (sensitivity), and a
shuffled-label null (5 permutations, labels shuffled within the pool) at the best readout.
    python vera_refit.py <tag> <state> [...]
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from vera_core import adir, CITY, EVF, HERE, LINE_T, PANEL, POOL, YF, pooled

TAG, STATES = sys.argv[1], sys.argv[2:]
CS = [0.01, 0.1, 1, 10]
pi = np.where(POOL)[0]


def fit(X, y, C):
    sc = StandardScaler().fit(X[pi]); lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    lr.fit(sc.transform(X[pi]), y[pi])
    return lambda Z: lr.decision_function(sc.transform(Z)), lr.n_iter_[0]


def oof(X, y, C):
    s = np.empty(len(pi))
    for tr, te in GroupKFold(5).split(pi, groups=CITY[pi]):
        sc = StandardScaler().fit(X[pi[tr]])
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(X[pi[tr]]), y[pi[tr]])
        s[te] = lr.decision_function(sc.transform(X[pi[te]]))
    return roc_auc_score(y[pi], s)


def catch(sc):
    t = np.sort(sc[LINE_T])[::-1][1]
    return float((sc[EVF] > t).mean()), sc > t


res = {}
for s in STATES:
    d = adir(s)
    T = np.load(d / "tok.npy", mmap_mode="r"); o = np.load(d / "off.npy")
    A = np.asarray(T[o[1:] - 2][:, PANEL], np.float32)
    rs = {}
    for j, r in enumerate(PANEL):
        X = A[:, j].astype(np.float64)
        aucs = {C: oof(X, YF, C) for C in CS}
        best = max(aucs.values())
        Cr = min(C for C in CS if aucs[C] >= best - 0.002)
        per_c = {}
        for C in CS:
            f, it = fit(X, YF, C)
            per_c[C] = {"catch": catch(f(X))[0], "n_iter": int(it)}
        rs[r] = {"oof": aucs, "C": Cr, "catch": per_c[Cr]["catch"], "per_C": per_c}
        print(s, r, "C", Cr, "oof", round(aucs[Cr], 4), "catch", round(per_c[Cr]["catch"], 3), flush=True)
    rb = max(rs, key=lambda r: rs[r]["oof"][rs[r]["C"]])
    X = A[:, PANEL.index(rb)].astype(np.float64)
    rng = np.random.default_rng(1022)
    null = []
    for k in range(5):
        y = YF.copy(); y[pi] = rng.permutation(YF[pi])
        f, _ = fit(X, y, rs[rb]["C"])
        null.append(catch(f(X))[0])
    res[s] = {"per_readout": rs, "best_readout": int(rb), "best_catch": rs[rb]["catch"],
              "max_catch_over_readouts": max(v["catch"] for v in rs.values()), "shuffled_null": null}
    print(s, "BEST", rb, rs[rb]["catch"], "max", res[s]["max_catch_over_readouts"], "null", null, flush=True)
    json.dump(res, open(HERE / f"vera_refit_{TAG}.json", "w"), indent=1)
