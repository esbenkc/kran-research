"""Which sd convention reproduces build's secondary R2 (F 0.4492, I 0.4364, Sigma 0.075)? Not gating."""
import json, numpy as np
from scipy.stats import norm
from vera_common import D, load, metrics, r2, r_P, test_set
L = load(); W = L["W"]; RB = json.load(open("rebuild/rebuild.json"))
sec = {m["name"]: m for m in json.load(open(D / "probes/secondary.json"))}
SW = np.load(D / "probes/secondary_weights.npz")["W"]
Ms, _ = metrics(L, 1.0, np.load(D / "fisher/fisher_big.npz")["F_common"])
ratios = {}
for c in RB["checks"]: ratios.setdefault(c["target"], []).append(c["my_delta_w"] / c["my_Gw"])
def sd_variants(su, y):
    a, b = su[y], su[~y]; n1, n0 = len(a), len(b)
    return {"pooled_ddof1": np.sqrt(((n1-1)*a.var(ddof=1)+(n0-1)*b.var(ddof=1))/(n1+n0-2)),
            "avgvar": np.sqrt((a.var()+b.var())/2), "total_sd": su.std()}
for key in ["pooled_ddof1", "avgvar", "total_sd"]:
    out = {}
    for k, P in list(Ms.items()) + [("shared", None)]:
        pred, meas = [], []
        for p in RB["secondary_pairs"]:
            t = p["target"]; w = W[L["name2i"][t]]; u = SW[sec[p["probe"]]["idx"]]
            X, y = test_set(L, t, 1); su, sw = X @ u, X @ w
            gap = su[y].mean() - su[~y].mean(); Gu = sw[y].mean() - sw[~y].mean(); sd = sd_variants(su, y)[key]
            r = 1.0 if P is None else r_P(P, u, w)
            pred.append(np.mean([norm.cdf(gap/sd/np.sqrt(2)) - norm.cdf((gap + q*r*Gu)/sd/np.sqrt(2)) for q in ratios[t]]))
            meas.append(p["meas"])
        out[k] = round(r2(pred, meas), 4)
    print(key, out)
