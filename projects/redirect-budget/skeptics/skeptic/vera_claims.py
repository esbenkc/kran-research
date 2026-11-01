"""Vera: rebuild the post's secondary claims from MY files.
- Spearman(RANK cap from the untrained model, measured logit swing after training) per run, with an exact-ish
  permutation null (20,000 shuffles of head labels).
- carriers: measured swing / RANK cap and / row-K cap (measured write size), TV / T_K.
- rank of the carrier (and of the top clamp head in undefined runs) by measured TV and by measured swing.
- carriers' RANK places, mean place share, and its permutation null (random 4 distinct heads).
"""
import json
import numpy as np
from scipy.stats import spearmanr

st = json.load(open("out/status.json"))
car = st["carriers"]
base = {"0.6B": json.load(open("out/r06.json"))["base"], "1.7B": json.load(open("out/r17_base.json"))}
rng = np.random.default_rng(1)
out = {}
for r, c in car.items():
    try:
        z = np.load(f"out/rows_{r}.npz")
    except FileNotFoundError:
        continue
    heads = list(z["heads"])
    rank = np.array([base[c["model"]][h]["rank"] for h in heads])
    osc = z["osc"].mean(1); tv = z["tv"].mean(1)
    rho = spearmanr(rank, osc).statistic
    null = np.array([spearmanr(rng.permutation(rank), osc).statistic for _ in range(2000)])
    h = heads.index(c["carrier"])
    out[r] = dict(defined=c["defined"], head=c["carrier"], spearman_rank_vs_swing=float(rho),
                  perm_p_two_sided=float((np.abs(null) >= abs(rho)).mean()),
                  carrier_place_by_tv=int((tv > tv[h]).sum()) + 1, carrier_place_by_swing=int((osc > osc[h]).sum()) + 1,
                  top_tv_head=heads[int(np.argmax(tv))],
                  swing=float(osc[h]), rank_cap=float(rank[h]), swing_over_rank_cap=float(osc[h] / rank[h]),
                  K_cap=float(z["K"].mean(1)[h]), swing_over_K_cap=float(osc[h] / z["K"].mean(1)[h]),
                  C_cap=float(z["C"].mean(1)[h]), swing_over_C_cap=float(osc[h] / z["C"].mean(1)[h]),
                  tv=float(tv[h]), T_K=float(z["T_K"].mean(1)[h]), tv_over_TK=float(tv[h] / z["T_K"].mean(1)[h]),
                  rank_place=int((rank > rank[h]).sum()) + 1, n_heads=len(heads))
for k, v in out.items():
    print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
# mean place share over distinct defined carriers, permutation null: 4 distinct heads (2 per model) drawn at random
pl = {}
for r, c in car.items():
    if c["defined"]:
        pl[(c["model"], c["carrier"])] = st["carrier_rank_position"][r]
n = {"0.6B": 32, "1.7B": 128}
share = np.mean([p / n[m] for (m, _), p in pl.items()])
sims = []
for _ in range(100000):
    s = []
    for m in n:
        k = sum(1 for (mm, _) in pl if mm == m)
        s += list(rng.choice(n[m], k, replace=False) + 1)  # places
    sims.append(np.mean([p / n[m] for p, m in zip(s, [m for m in n for _ in range(sum(1 for (mm, _) in pl if mm == m))])]))
sims = np.array(sims)
out["_places"] = dict(distinct=[f"{m}:{h}={p}/{n[m]}" for (m, h), p in pl.items()], mean_place_share=float(share),
                      P_random_le=float((sims <= share).mean()), P_random_ge=float((sims >= share).mean()))
print(out["_places"])
json.dump(out, open("out/claims.json", "w"), indent=1)
