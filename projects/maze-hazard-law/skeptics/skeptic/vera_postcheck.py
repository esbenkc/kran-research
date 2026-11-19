#!/usr/bin/env python3
"""Check the draft's foldout numbers that vera_rebuild.py does not print: Pearson statistic, run-level bootstrap
intervals of h_all and R, Weibull-k profile interval, and the dead-end 'same page every call' loop count.
usage: vera_postcheck.py <data root>"""
import collections, glob, json, math, sys
import numpy as np
from scipy import stats
from vera_rebuild import load_runs, events, decisions, law, weib_ll, mpred
root = sys.argv[1]
rng = np.random.default_rng(7)
for agent in ("qwen3-8b-nothink", "gpt-oss-20b-low"):
    evs = [e for e in (events(d) for d in load_runs(f"{root}/{agent}", "maze")) if e["entered"]]
    L = law(evs)
    rows = decisions(evs); h = L["h_all"]
    chi = sum((g - e*h)**2/(e*h) + ((e-g) - e*(1-h))**2/(e*(1-h)) for e, g in L["bin_counts"])
    hs, Rs = [], []
    for _ in range(2000):
        b = [evs[i] for i in rng.integers(0, len(evs), len(evs))]
        r = decisions(b); E = len(r); G = sum(x for _, x in r)
        hs.append(G/E)
        E3 = sum(1 for p, _ in r if p <= 3); G3 = sum(x for p, x in r if p <= 3)
        Lb = None
        if G3:
            from vera_rebuild import km_median
            m, _ = km_median(b)
            if m: Rs.append(mpred(G3/E3, L["K"]) / m)
    P = np.array([p for p, _ in rows], float); X = np.array([x for _, x in rows], float)
    # profile likelihood for k
    from scipy import optimize
    def prof(k):
        r = optimize.minimize_scalar(lambda z: -weib_ll((1/(1+math.exp(-z)), k), P, X), bounds=(-5, 30), method="bounded")
        return -r.fun
    ks = np.exp(np.linspace(math.log(0.2), math.log(8), 400)); ll = np.array([prof(k) for k in ks])
    ok = ks[ll >= ll.max() - stats.chi2.ppf(0.95, 1)/2]
    print(agent, f"chi {chi:.1f} df {len(L['bin_counts'])-1}", "h_all boot", np.percentile(hs, [2.5, 97.5]).round(3),
          "R boot", np.percentile(Rs, [2.5, 97.5]).round(2) if Rs else None, f"(defined {len(Rs)}/2000)",
          "k", round(ks[ll.argmax()], 2), "profile CI", (round(ok.min(), 2), round(ok.max(), 2)))
# dead-end loops
for f in sorted(glob.glob(f"{root}/qwen3-8b-nothink/raw/deadend/*.json")):
    d = json.load(open(f))
    if d["end"] != "cap": continue
    c = collections.Counter(a.get("path") for a in d["actions"] if a.get("tool") == "fetch")
    canary = d["site"]["canary_path"]
    first = min(a["step"] for a in d["actions"] if a.get("path") == canary)
    after = collections.Counter(a.get("path") for a in d["actions"] if a["step"] > first)
    print("deadend", f[-8:], "steps", d["steps_used"], "entry step", first, "after-entry paths", dict(after), "parse_fail", d["parse_fail"])
