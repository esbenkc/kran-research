"""Vera: city bootstrap of the pooled PRICE ratios P/U and R/U (D1, horizon 150) from raw ckpt npz.
Resample the 298 eval cities with replacement (each city brings its false and true eval rows); the re-set line stays
the 2nd-highest of the 200 line-setting true (not resampled). 2000 draws. Also the seed-swap check: every seed pairing."""
import json, numpy as np, itertools
S = json.load(open("data/ref/split.json")); sp = np.array([e["split"] for e in S]); tr = np.array([e["true"] == 1 for e in S])
city = np.array([e["city"] for e in S]); LS = (sp == "refit") & tr; ev = np.where(sp == "eval")[0]
cities = np.unique(city[ev]); rows_of = {c: ev[city[ev] == c] for c in cities}
runs = {a: [f"data/runs/{a}__lam10000__seed{s}" for s in (0, 1)] for a in ("unproj", "proj", "rand")}
col = {"unproj": "unproj", "proj": "proj", "rand": "rand"}
pre = {}
for a, rs in runs.items():
    for r in rs:
        for t in (50, 100, 150):
            d = np.load(f"{r}/ckpt_{t}.npz", allow_pickle=True); names = [str(x) for x in d["probes"]]
            s = d["scores"][:, names.index(col[a])].astype(np.float64); line = np.sort(s[LS])[::-1][1]
            kl = d["kl"].astype(np.float64); ko = d["koff"]; cp = d["cpos"]
            klc = np.array([kl[ko[i] + cp[i]] for i in range(len(S))])
            pre[(r, t)] = ((s > line) & ~tr, klc)
def price(a, idx, seeds=(0, 1)):
    best = np.inf
    for t in (50, 100, 150):
        fl = idx[~tr[idx]]
        catch = np.mean([pre[(runs[a][s], t)][0][fl].mean() for s in seeds])
        k = np.mean([pre[(runs[a][s], t)][1][idx].mean() for s in seeds])
        if catch <= 0.2: best = min(best, k)
    return best
full = ev
base = {a: price(a, full) for a in runs}
print("point", base, "P/U %.3f R/U %.3f" % (base["proj"] / base["unproj"], base["rand"] / base["unproj"]))
rng = np.random.default_rng(0); B = []
for b in range(2000):
    idx = np.concatenate([rows_of[c] for c in rng.choice(cities, len(cities))])
    p = {a: price(a, idx) for a in runs}; B.append((p["proj"] / p["unproj"], p["rand"] / p["unproj"]))
B = np.array(B)
print("P/U 95%% [%.3f, %.3f]  R/U 95%% [%.3f, %.3f]  P(P/U>=1.5) %.4f  P(P/U>=3) %.4f" % (*np.percentile(B[:, 0], [2.5, 97.5]), *np.percentile(B[:, 1], [2.5, 97.5]), (B[:, 0] >= 1.5).mean(), (B[:, 0] >= 3).mean()))
pairs = {}
for sp_, su in itertools.product((0, 1), (0, 1)):
    pairs[f"P s{sp_} / U s{su}"] = price("proj", full, (sp_,)) / price("unproj", full, (su,))
print("single-seed ratios", {k: round(v, 3) for k, v in pairs.items()})
json.dump(dict(point=base, boot_PU=np.percentile(B[:, 0], [2.5, 50, 97.5]).tolist(), boot_RU=np.percentile(B[:, 1], [2.5, 50, 97.5]).tolist(), single_seed=pairs), open("skeptic/vera_boot.json", "w"), indent=1)
