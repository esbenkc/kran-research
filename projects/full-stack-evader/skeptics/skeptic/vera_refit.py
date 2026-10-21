"""Vera: re-fit recovery (PREREG s8) with own code. For each attacked state: country-token linear probe re-fit on THAT
state's 1400 pool statements at readouts 1..29 (C from C.json, i.e. the frozen rule's output), line from that state's
200 line-setting true statements. Best readout per state = highest 5-fold GroupKFold(city) out-of-fold AUROC on the pool
(no eval data). Recovery = pooled eval catch at the per-state best readout; max over readouts reported; shuffled-label
null (5 permutations, pool labels permuted within nothing, seed 1021+k) at the best readout.
    python vera_refit.py <tag> <state> [...]
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from vera_core import CITY, HERE, POOL, PROJ, YF, auroc, flags, pooled

TAG, STATES = sys.argv[1], sys.argv[2:]
C = json.load(open(PROJ / "data/fits/C.json"))
PI = np.where(POOL)[0]


def fit_score(X, y, c):
    sc = StandardScaler().fit(X[PI])
    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(sc.transform(X[PI]), y[PI])
    return lr.decision_function(sc.transform(X))


def oof(X, c):
    s = np.empty(len(PI))
    for tr, te in GroupKFold(5).split(PI, YF[PI], CITY[PI]):
        sc = StandardScaler().fit(X[PI[tr]])
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(sc.transform(X[PI[tr]]), YF[PI[tr]])
        s[te] = lr.decision_function(sc.transform(X[PI[te]]))
    return auroc(s, YF[PI])


res = {}
best_flags, maxl = [], {}
per_layer_flags = {r: [] for r in range(1, 30)}
for s in STATES:
    Fc = np.load(HERE / f"feats/{s}_ctry.npy", mmap_mode="r")
    rr = {}
    for r in range(1, 30):
        X = np.asarray(Fc[:, r], np.float64); c = C[f"tok{r}"]["C"]
        sco = fit_score(X, YF, c)
        f = flags(sco)
        per_layer_flags[r].append(f)
        rr[r] = {"oof": oof(X, c), "catch": pooled([f])["catch"]}
    best = max(rr, key=lambda r: rr[r]["oof"])
    X = np.asarray(Fc[:, best], np.float64); c = C[f"tok{best}"]["C"]
    best_flags.append(flags(fit_score(X, YF, c)))
    nulls = []
    for k in range(5):
        y = YF.copy(); y[PI] = np.random.default_rng(1021 + k).permutation(YF[PI])
        nulls.append(pooled([flags(fit_score(X, y, c))])["catch"])
    res[s] = {"best": best, "best_catch": rr[best]["catch"], "null": nulls, "layers": rr}
    print(s, "best", best, rr[best], "null", nulls, flush=True)
res["pooled_best"] = pooled(best_flags)
res["pooled_by_layer"] = {r: pooled(per_layer_flags[r]) for r in range(1, 30)}
print("pooled recovery at best", res["pooled_best"])
print("pooled by layer", {r: round(v["catch"], 3) for r, v in res["pooled_by_layer"].items()})
json.dump(res, open(HERE / f"vera_refit_{TAG}.json", "w"), indent=1, default=float)
