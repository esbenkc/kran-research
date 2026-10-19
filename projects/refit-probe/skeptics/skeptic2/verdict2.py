"""Hugo (skeptic2): verdict robustness. Reuses only my own check.py.

(1) main gaps (base minus policy, refit AUROC at n=200, 20 draws) under 3 draw seeds;
(2) robustness guard: the same gaps on the TEST answers where judge == string check,
    (a) refit on judge labels, scored on the clean test subset; (b) refit on clean pool answers only.
Paired bootstrap over the 400 shared test prompts, 1000 resamples.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from check import load, fit_score, OUT

POL = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
data = {p: load(p) for p in POL}


def scores(d, seed, clean_train=False, k=20, n=200):
    rng = np.random.default_rng(seed)
    pool = np.where(d["split"] == "pool")[0]
    ok = ~np.isnan(d["y"][pool])
    if clean_train:
        ok &= d["ys"][pool] == d["y"][pool]
    pool = pool[ok]
    h, dd = pool[d["y"][pool] == 0], pool[d["y"][pool] == 1]
    S = []
    for _ in range(k):
        f = np.concatenate([rng.choice(h, n // 2, replace=False), rng.choice(dd, n // 2, replace=False)])
        S.append(fit_score(d["own"][f], d["y"][f].astype(int), d["own"][:400])[0])
    return np.array(S)  # (k, 400) scores on all 400 test rows


def auc_rows(d, S, rows, clean):
    y = d["y"][rows]
    m = ~np.isnan(y)
    if clean:
        m &= d["ys"][rows] == y
    y = y[m].astype(int)
    if y.min() == y.max():
        return np.nan
    return float(np.mean([roc_auc_score(y, s[rows][m]) for s in S]))


def gaps(Sd, clean):
    out = {}
    rng = np.random.default_rng(1019)
    B = [rng.integers(0, 400, 400) for _ in range(1000)]
    full = np.arange(400)
    for p in POL[1:]:
        g = auc_rows(data["base"], Sd["base"], full, clean) - auc_rows(data[p], Sd[p], full, clean)
        bs = [auc_rows(data["base"], Sd["base"], b, clean) - auc_rows(data[p], Sd[p], b, clean) for b in B]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out[p] = [round(g, 4), round(lo, 4), round(hi, 4)]
    return out


def verdict(G):
    if all(v[0] <= 0.05 for v in G.values()):
        return "YES"
    if any(v[0] > 0.05 and v[1] > 0 for v in G.values()):
        return "NO"
    return "UNCONFIRMED"


R = {}
for seed in [11, 22, 33]:
    Sd = {p: scores(data[p], seed) for p in POL}
    G = gaps(Sd, clean=False)
    R[f"main_seed{seed}"] = {"gaps[pt,lo,hi]": G, "verdict": verdict(G)}
    if seed == 11:
        G = gaps(Sd, clean=True)
        R["clean_eval_only"] = {"gaps[pt,lo,hi]": G, "verdict": verdict(G)}
Sd = {p: scores(data[p], 11, clean_train=True) for p in POL}
G = gaps(Sd, clean=True)
R["clean_train_and_eval"] = {"gaps[pt,lo,hi]": G, "verdict": verdict(G)}
json.dump(R, open(OUT / "verdict2_out.json", "w"), indent=1)
print(json.dumps(R, indent=1))
