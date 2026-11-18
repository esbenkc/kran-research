"""Vera: null tests for the gpt-5-nano side-row verdict (FAILS by shape), using my own rebuild (vera_rebuild.py).

N1 parametric bootstrap: a constant-hazard agent at the observed h_all, same n entered runs, same reachable depth,
   no lost exits -> how often do chi p, Weibull k, Weibull p and R reach the observed values by luck?
N2 per seed block: Weibull k and LR p inside each block (does the rising shape replicate across blocks?).
N3 cap-only null: the same runs with every exit turned into a cap -> the rule must say UNTESTED, never HOLDS/FAILS.
N4 shuffled pilot: draw the 3 'pilot' pages at random positions (3 random consecutive pages from 1..10) -> is the
   pilot miss specific to pages 1-3?
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import vera_rebuild as vr

data = Path(sys.argv[1])
agent = sys.argv[2] if len(sys.argv) > 2 else "gpt-5-nano"
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
res = vr.agent_summary(data / agent)
rows = res["maze"]["_rows"]
obs = vr.law(rows)
out = {"observed": {k: obs[k] for k in ("n", "G", "G3", "E3", "h3", "h_all", "km_median", "pred_median", "R", "chi_p",
                                         "weib_p", "weib_k", "verdict", "why")}}
print("observed", out["observed"])

# N1
rng = np.random.default_rng(20261118)
h = obs["h_all"]
n = obs["n"]
Ks = [81 - r["k"] for r in rows]
cnt = dict(chi=0, wk=0, wp=0, R=0, fails=0, holds=0)
ks = []
for _ in range(reps):
    sim = []
    for K in Ks:
        D = int(rng.geometric(h))
        sim.append(dict(D=D, cause="g", k=81 - K) if D <= K else dict(D=K, cause="c", k=81 - K))
    L = vr.law(sim)
    ks.append(L["weib_k"] or float("nan"))
    cnt["chi"] += L["chi_p"] is not None and L["chi_p"] <= obs["chi_p"]
    cnt["wk"] += L["weib_k"] is not None and L["weib_k"] >= obs["weib_k"]
    cnt["wp"] += L["weib_p"] is not None and L["weib_p"] <= obs["weib_p"]
    cnt["R"] += L["R"] is not None and L["R"] >= obs["R"]
    cnt["fails"] += L["verdict"] == "FAILS"
    cnt["holds"] += L["verdict"] == "HOLDS"
out["N1_parametric_bootstrap"] = {k: v / reps for k, v in cnt.items()} | {
    "reps": reps, "h": h, "weib_k_null_q95": float(np.nanquantile(ks, 0.95)), "weib_k_null_median": float(np.nanmedian(ks))}
print("N1", out["N1_parametric_bootstrap"])

# N2
blocks = {}
for b in sorted({r["block"] for r in rows}):
    rb = [r for r in rows if r["block"] == b]
    d, e = vr.risk(rb)
    p, k = vr.weibull_lr(d, e)
    pc, B = vr.pearson(d, e)
    blocks[b] = dict(n=len(rb), weib_k=k, weib_p=p, chi_p=pc)
out["N2_blocks"] = blocks
print("N2", blocks)

# N3
capped = [dict(r, cause="c", D=81 - r["k"]) for r in rows]
L3 = vr.law(capped)
out["N3_cap_only"] = dict(verdict=L3["verdict"], why=L3.get("why"))
print("N3", out["N3_cap_only"])

# N4: hazard on 3 consecutive pages starting at s, prediction vs observed median
d, e = vr.risk(rows)
med = obs["km_median"]
n4 = {}
for s in range(1, 9):
    G, E = e[s:s + 3].sum(), d[s:s + 3].sum()
    hh = G / E if E else float("nan")
    pm = vr.pred_median(hh, obs["K"]) if E else float("nan")
    n4[f"pages {s}-{s + 2}"] = dict(G=int(G), E=int(E), h=round(hh, 4), pred=round(pm, 2), R=round(pm / med, 3))
out["N4_window_hazards"] = n4
print("N4", json.dumps(n4))
Path(f"nulls_{agent}.json").write_text(json.dumps(out, indent=1, default=float))
