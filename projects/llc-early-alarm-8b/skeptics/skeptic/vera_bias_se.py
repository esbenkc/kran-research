"""Vera: miss in units of the estimator's own batch-means SE, rep 0 and all reps, from data/gate/grid.json raw draws."""
import json, math, sys, numpy as np
G = json.load(open(sys.argv[1])); nb = G["n"] / math.log(G["n"]); true = {1: 236.0, 8: 312.0}
out = {0: [], "all": []}
for row in G["rows"]:
    d = np.array(row["losses"], float)[:, G["burn"]:].reshape(-1)
    lam = nb * (d.mean() - row["L_wstar"])
    bm = np.array([nb * (b.mean() - row["L_wstar"]) for b in d.reshape(5, -1)]); se = bm.std(ddof=1) / math.sqrt(5)
    if np.isnan(lam): continue
    z = abs(lam - true[row["r"]]) / se
    out["all"].append((z, row["c"], row["gamma"], row["r"], row["rep"], lam, se))
    if row["rep"] == 0: out[0].append((z, row["c"], row["gamma"], row["r"], lam, se))
for k in (0, "all"):
    zs = sorted(out[k])
    print(f"{k}: finished {len(zs)}; min miss {zs[0][0]:.2f} SE at {zs[0][1:]}; count < 5 SE: {sum(z[0] < 5 for z in zs)}; < 3 SE: {sum(z[0] < 3 for z in zs)}")
best = [x for x in out[0] if x[1] == 0.3 and x[2] == 10.0]
print("c0.3 g10 rep0:", [(r, round(l, 1), round(s, 2), round(z, 2)) for z, c, g, r, l, s in best])
reps = [x for x in out["all"] if x[1] == 0.3 and x[2] == 10.0]
for r in (1, 8):
    v = [x[5] for x in reps if x[3] == r]; print(f"c0.3 g10 r{r} reps: {[round(x,1) for x in sorted(v)]} span {100*(max(v)-min(v))/np.mean(v):.1f}% of mean")
