"""Vera: max-panel baseline from 1021 raw acts (clean, layer-18 attack a s0-s2, lambda0).
Country-token acts gathered from tok.npy (row off[i+1]-2), scored with the frozen panel.npz weights by my own arithmetic,
compared to build's saved scores npz. Then per-readout catch and max-panel catch, own line per scored model.
    python vera_baseline.py
"""
import json

import numpy as np

from vera_core import HERE, OLD, PANEL, YF, EVAL, auroc, flags, maxpanel, pooled, zmoments

STATES = {"clean": ["clean"], "L18": [f"a__gap__lam10000__seed{i}" for i in range(3)],
          "L18_lam0": ["a__gap__lam0__seed0"], "mega": [f"b__gap300__lam10000__seed{i}" for i in range(3)]}
BP = np.load(OLD / "data/fits/panel.npz")


def ctry(state):
    T = np.load(OLD / f"data/acts/{state}/tok.npy", mmap_mode="r")
    o = np.load(OLD / f"data/acts/{state}/off.npy")
    return np.asarray(T[o[1:] - 2][:, PANEL], np.float64)  # [2992, 20, 1536]


def my_scores(state):
    X = ctry(state)
    out = {}
    for j, r in enumerate(PANEL):
        mu, sd, w, b = (BP[f"tok{r}__{k}"].astype(np.float64) for k in ["mu", "sd", "w", "b"])
        out[r] = ((X[:, j] - mu) / sd) @ w + b
    return out


S = {}
maxdiff = {}
for g, sts in STATES.items():
    for s in sts:
        S[s] = my_scores(s)
        bs = np.load(OLD / f"data/scores/{s}.npz")
        maxdiff[s] = max(float(np.abs(S[s][r] - bs[f"tok{r}"]).max()) for r in PANEL)
        print(s, "max|mine-build| over panel", maxdiff[s], flush=True)
mom = zmoments(S["clean"])
res = {"maxdiff_vs_build_scores": maxdiff, "groups": {}}
for g, sts in STATES.items():
    d = {"per_readout": {}, "max_all20": None}
    for r in PANEL:
        d["per_readout"][r] = pooled([flags(S[s][r]) for s in sts])
    for name, ro in [("max_all20", PANEL), ("max_10_28", PANEL[:-1])]:
        m = [maxpanel(S[s], mom, ro) for s in sts]
        d[name] = pooled([flags(x) for x in m])
        d[name + "_auroc"] = [auroc(x[EVAL], YF[EVAL]) for x in m]
    res["groups"][g] = d
    print(g, "max20", d["max_all20"], "max10-28", d["max_10_28"])
    print("   per-readout", {r: round(v["catch"], 3) for r, v in d["per_readout"].items()})
json.dump(res, open(HERE / "vera_baseline.json", "w"), indent=1)
