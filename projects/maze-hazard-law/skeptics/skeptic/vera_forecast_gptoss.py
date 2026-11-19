#!/usr/bin/env python3
"""Forecast, before the full-prefill gpt-oss-20b data exist, what the frozen rule will say on 60 maze runs.

Fits the discrete Weibull (q, k) on the 9 reuse-on pilot runs (a preview, not status data), draws 60 entered runs from
it (reach 78 pages, the pilot's median 81 - k_r), and applies vera_rebuild.law(). Also a fit-uncertainty version:
each replicate first draws (q, k) from a parametric bootstrap of the 9 runs.
usage: vera_forecast_gptoss.py gptoss_reuseON_partial.json [reps]
"""
import collections
import json
import math
import sys

import numpy as np
from scipy import optimize

from vera_rebuild import decisions, law, weib_ll

R = json.load(open(sys.argv[1]))
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
runs = [r for r in R["per_run"] if r.get("D") is not None]
rng = np.random.default_rng(1119)


def fit(rows):
    P = np.array([p for p, _ in rows], float)
    X = np.array([x for _, x in rows], float)
    best = None
    for k0 in (0.8, 1.5, 3.0, 5.0):
        for q0 in (0.9, 0.99, 0.999):
            r = optimize.minimize(lambda z: -weib_ll((1 / (1 + math.exp(-z[0])), math.exp(z[1])), P, X),
                                  [math.log(q0 / (1 - q0)), math.log(k0)], method="Nelder-Mead",
                                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    return 1 / (1 + math.exp(-best.x[0])), math.exp(best.x[1])


def draw(q, k, n, reach=78):
    evs = []
    for _ in range(n):
        d, kind = 0, "censored"
        while d < reach:
            d += 1
            h = 1 - q ** (d ** k - (d - 1) ** k)
            if rng.random() < h:
                kind = "giveup"
                break
        evs.append({"D": d, "kind": kind, "k": 81 - reach, "end": "answer" if kind == "giveup" else "cap"})
    return evs


q, k = fit(decisions(runs))
out = {"fit_on_9_runs": {"q": q, "k": k}}
for label, boot in (("fixed_fit", False), ("bootstrap_fit", True)):
    c = collections.Counter()
    g3 = []
    for _ in range(reps):
        qq, kk = (fit(decisions(draw(q, k, len(runs)))) if boot else (q, k))
        L = law(draw(qq, kk, 60))
        c[L["verdict"].split(" (")[0] if not L["verdict"].startswith("UNTESTED") else L["verdict"]] += 1
        g3.append(L["G3"])
    out[label] = {"verdicts": {v: n / reps for v, n in c.items()}, "P(G3>=1)": float(np.mean(np.array(g3) >= 1)),
                  "G3_median": float(np.median(g3))}
print(json.dumps(out, indent=1))
