#!/usr/bin/env python3
"""Null test for a REJECT by shape: can luck alone fail a TRUE constant hazard at the observed size?

Simulates entered maze runs under a geometric give-up law at the agent's fitted h_all (same n, same per-run reachable
depth 81 - k_r from the observed runs, same lost rate per page), feeds them through vera_rebuild.law() (the frozen rule),
and reports P(FAILS shape), P(FAILS any), and how often the simulated Pearson / Weibull p-values reach the observed ones.

usage: vera_null.py <vera_rebuild json> [reps]
"""
import json
import sys

import numpy as np

from vera_rebuild import law

R = json.load(open(sys.argv[1]))
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
L = R["law_giveup"]
h, n = L["h_all"], L["n"]
runs = [r for r in R["per_run"] if r.get("D") is not None]
reach = [81 - r["k"] for r in runs]
lost_rate = L["lost"] / max(1, L["E"])
rng = np.random.default_rng(20261119)
cnt = {"FAILS (shape)": 0, "fails_any": 0, "holds": 0, "untested": 0}
p_le_obs_a = p_le_obs_b = 0
for _ in range(reps):
    evs = []
    for i in range(n):
        cap = reach[i % len(reach)]  # page budget: at most one new page per call
        d, kind = 0, "censored"
        while d < cap:
            d += 1
            u = rng.random()
            if u < h:
                kind = "giveup"
                break
            if u < h + lost_rate:
                kind = "lost"
                break
        evs.append({"D": d, "kind": kind, "k": 81 - cap, "end": "cap" if kind == "censored" else "answer"})
    r = law(evs)
    v = r["verdict"]
    if v.startswith("FAILS (shape)"):
        cnt["FAILS (shape)"] += 1
    if v.startswith("FAILS"):
        cnt["fails_any"] += 1
    elif v == "HOLDS":
        cnt["holds"] += 1
    else:
        cnt["untested"] += 1
    if r["p_pearson"] is not None and r["p_pearson"] <= L["p_pearson"]:
        p_le_obs_a += 1
    if r["p_weibull"] is not None and r["p_weibull"] <= L["p_weibull"]:
        p_le_obs_b += 1
out = {"agent": R["agent"], "h_all": h, "n": n, "reps": reps,
       **{k: v / reps for k, v in cnt.items()},
       "P(pearson p <= observed %.2e)" % L["p_pearson"]: p_le_obs_a / reps,
       "P(weibull p <= observed %.2e)" % L["p_weibull"]: p_le_obs_b / reps}
print(json.dumps(out, indent=1))
