"""Vera: pooled base-arm cells (R1,R3,R4) with a joint city-cluster bootstrap (same city draw for every seed), own fits.
Monitors as vera_state (F1 C-grid OOF, F2 best-OOF readout) and vera_m1 (C.json). Pooled = mean of per-seed catch."""
import json, numpy as np
import vera_state as V
Cj = json.load(open(V.P + "data/fits/C.json"))
runs = ["refit50__lam10000__seed0", "refit50__lam10000__seed1", "refit50__lam10000__seed2"]
f = V.EV & (V.y == 1)
cities = np.unique(V.city[f]); byc = {c: np.where(f & (V.city == c))[0] for c in cities}
hits = {"M1": [], "F1": [], "F2": []}; auc = {"M1": [], "F1": [], "F2": []}
for r in runs:
    A = np.load(V.P + f"data/ctok/{r}.npy", mmap_mode="r")
    m1, f1, oo = [], [], []
    for j, R in enumerate(V.READ):
        X = np.asarray(A[:, j, :], np.float64)
        m1.append(V.fit(X, V.ATT, V.y, Cj[f"tok{R}"]["C"]))
        C, a = V.pick_C(X, V.TR, V.y); f1.append(V.fit(X, V.TR, V.y, C)); oo.append(a)
    j2 = int(np.argmax(np.round(oo, 12)))
    for k, s in (("M1", V.zmax(m1)), ("F1", V.zmax(f1)), ("F2", f1[j2])):
        t = V.line(s[V.LINE]); hits[k].append(s > t); auc[k].append(V.stats(s, boot=False)["auroc"])
rng = np.random.default_rng(1021)
out = {}
for k in hits:
    per = [float(h[f].mean()) for h in hits[k]]
    bs = []
    for _ in range(2000):
        ii = np.concatenate([byc[c] for c in rng.choice(cities, len(cities))])
        bs.append(np.mean([h[ii].mean() for h in hits[k]]))
    out[k] = dict(per_seed=per, pooled=float(np.mean(per)), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  auroc_per_seed=auc[k], auroc_pooled=float(np.mean(auc[k])))
kl = {r: json.load(open(f"vera_statekl_{n}_bf16mps.json"))["kl_tok_mean"] for r, n in zip(runs, ["R1", "R3", "R4"])}
P = 0.006145047
out["kl_bf16_per_seed"] = kl; out["kl_pooled"] = float(np.mean(list(kl.values()))); out["kl_over_P"] = out["kl_pooled"] / P
out["clauses"] = {"1_fresh_lt30": out["F1"]["pooled"] < 0.30 and out["F2"]["pooled"] < 0.30, "2_M1_lt20": out["M1"]["pooled"] < 0.20,
                  "3_kl_le_0.0096": out["kl_pooled"] <= 0.0096, "4_in_band": P / 3 <= out["kl_pooled"] <= 3 * P}
json.dump(out, open("vera_pool.json", "w"), indent=1)
print(json.dumps(out, indent=1))
