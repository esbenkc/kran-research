"""Why does the prompt-bootstrap upper bound equal the point slope? Per-run distribution of resampled T."""
import json, glob, math, sys, numpy as np
R = sys.argv[1]; seeds = [5, 6, 7]
S_K = [0.05469028438159526 / 2 ** k for k in range(6)]
PP = {}
for s in seeds:
    for k in range(6):
        f = f"{R}/m8_k{k}_s{s}.json"
        try: d = json.load(open(f))
        except FileNotFoundError: continue
        ev = {}
        for c in d["dev_curve"] + d["dense_curve"]:
            ev[c["step"]] = (np.array(c["p_off"]), np.array(c["p_on"]))
        st = sorted(ev); PP[(s, k)] = (np.array(st), np.stack([ev[x][0] for x in st]), np.stack([ev[x][1] for x in st]), d["T_dense"])
rng = np.random.default_rng(0)
Tb = {key: [] for key in PP}
for _ in range(2000):
    idx = rng.integers(0, 64, 64)
    for key, (st, Po, Pn, T) in PP.items():
        g = (Po[:, idx] - Pn[:, idx]).mean(1); D = g - g[0]
        ok = np.where((D[:-1] >= .1) & (D[1:] >= .1))[0]
        Tb[key].append(st[ok[0]] if len(ok) else -1)
for key in sorted(PP):
    v = np.array(Tb[key]); T = PP[key][3]
    print(key, "T", T, "P(<T) %.3f P(=T) %.3f P(>T) %.3f" % ((v < T).mean(), (v == T).mean(), (v > T).mean()), "cens", (v == -1).sum(), "q2.5/97.5", np.quantile(v, [.025, .975]))
