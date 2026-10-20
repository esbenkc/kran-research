"""Vera: remaining nulls from my spec (#3746). (a) lin_14 and lin_18 refit with 5 shuffled label permutations (seeds 1..5):
clean eval AUROC and pooled attacked catch. (b) 200 random unit directions in the standardised mean-pooled space at
layers 8/14/18: twin-paired d' (clean spread units), clean and attacked, vs the fitted probe's d'.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_nulls.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vera_core import ATT, EV_IDX, FE, HERE, POOL, STATES, YF, auroc, dprime, flags, pooled_catch

M = {s: np.load(FE / f"{s}_mean.npy", mmap_mode="r") for s in STATES}
res = {}
for l in [8, 14, 18]:
    X = {s: np.asarray(M[s][:, l], np.float64) for s in STATES}
    sc = StandardScaler().fit(X["clean"][POOL]); Z = {s: sc.transform(X[s]) for s in STATES}
    if l in (14, 18):
        sh = []
        for k in range(1, 6):
            yp = YF.copy(); yp[POOL] = np.random.default_rng(k).permutation(YF[POOL])
            lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(Z["clean"][POOL], yp[POOL])
            S = {s: lr.decision_function(Z[s]) for s in STATES}
            sh.append({"clean_auroc": auroc(S["clean"][EV_IDX], YF[EV_IDX]), "att_catch": pooled_catch([flags(S[s]) for s in ATT])})
        res[f"shuffle_L{l}"] = sh
        print(f"L{l} shuffled: clean AUROC", [round(x["clean_auroc"], 3) for x in sh], "att catch", [round(x["att_catch"], 3) for x in sh], flush=True)
    rng = np.random.default_rng(7)
    D = rng.standard_normal((200, Z["clean"].shape[1])); D /= np.linalg.norm(D, axis=1, keepdims=True)
    dc, da = [], []
    for d in D:
        S = {s: Z[s] @ d for s in STATES}
        dp = dprime(S, S["clean"]); dc.append(dp["clean"][0]); da.append(np.mean([dp[s][0] for s in ATT]))
    dc, da = np.abs(dc), np.abs(da)
    res[f"randdir_L{l}"] = {"clean_absd_p50_p95_max": np.percentile(dc, [50, 95, 100]).tolist(),
                            "att_absd_p50_p95_max": np.percentile(da, [50, 95, 100]).tolist()}
    print(f"L{l} random dirs |d'| clean p50/p95/max", np.round(res[f"randdir_L{l}"]["clean_absd_p50_p95_max"], 3).tolist(),
          "attacked", np.round(res[f"randdir_L{l}"]["att_absd_p50_p95_max"], 3).tolist(), flush=True)
json.dump(res, open(HERE / "vera_nulls.json", "w"), indent=1, default=float)
