"""Vera: the scout's prediction C1-C4 (PREREG sec 6) and validity rows Q/K/C, from MY rows_<run>.npz + carriers.
U_h = mean over prompts of the measured value, B_h = mean over prompts of the per-prompt bound; under iff U_h <= B_h.
Output: out/scout.json
"""
import json, itertools
import numpy as np

st = json.load(open("out/status.json"))
car = st["carriers"]
runs = [r for r in car if __import__("os").path.exists(f"out/rows_{r}.npz")]
res = {"runs": runs}
for space, meas, pre in [("tv", "tv", "T_"), ("logit", "osc", "")]:
    for row in "SQKC":
        under, tot, cells_u, cells_t, maxratio = 0, 0, 0, 0, 0
        car_rows = {}
        for r in runs:
            z = np.load(f"out/rows_{r}.npz")
            U = z[meas].mean(1); Bd = z[pre + row].mean(1)
            under += int((U <= Bd).sum()); tot += len(U)
            cells_u += int((z[meas] <= z[pre + row]).sum()); cells_t += z[meas].size
            maxratio = max(maxratio, float((z[meas] / np.maximum(z[pre + row], 1e-12)).max()))
            heads = list(z["heads"])
            if car[r]["defined"]:
                h = heads.index(car[r]["carrier"])
                car_rows[r] = dict(head=car[r]["carrier"], U=float(U[h]), B=float(Bd[h]), ratio_B_over_U=float(Bd[h] / U[h]),
                                   rank_by_B=int((Bd > Bd[h]).sum()) + 1)
        res[f"{row}_{space}"] = dict(heads_under=under, heads_total=tot, share_heads=under / tot,
                                      share_cells=cells_u / cells_t, max_cell_ratio=maxratio, carriers=car_rows)
# C1-C4, row S in TV space
S = res["S_tv"]
C1 = S["share_heads"] >= 0.95
C2 = all(c["U"] <= c["B"] <= 10 * c["U"] for c in S["carriers"].values())
hits3 = 0
pools = {}
for r in runs:
    z = np.load(f"out/rows_{r}.npz")
    Bd = z["T_S"].mean(1); heads = list(z["heads"])
    top = [heads[i] for i in np.argsort(-Bd, kind="stable")[:3]]
    if car[r]["defined"] and car[r]["carrier"] in top:
        hits3 += 1
    pools.setdefault(car[r]["model"], heads)
C3 = hits3 >= 5
# C4: 50 random 3-sets per model, numpy default_rng(0), one set per model scored on all its runs; plus exact
rng = np.random.default_rng(0)
cnt = 0
for _ in range(50):
    sel = {m: set(rng.choice(p, 3, replace=False)) for m, p in pools.items()}
    h = sum(1 for r in runs if car[r]["defined"] and car[r]["carrier"] in sel[car[r]["model"]])
    cnt += h >= hits3
res["scout"] = dict(C1=C1, C1_share=S["share_heads"], C2=C2, C3_hits=hits3, C3=C3, C4_frac_ge=cnt / 50,
                    C4=(cnt / 50) <= 0.05, held=bool(C1 and C2 and C3 and (cnt / 50) <= 0.05))
json.dump(res, open("out/scout.json", "w"), indent=1)
for k, v in res.items():
    if k != "runs":
        print(k, json.dumps({kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v)[:700])
