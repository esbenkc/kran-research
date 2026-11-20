#!/usr/bin/env python3
"""Vera (skeptic), night 2026-11-20: robustness and null tests on the full-prefill status rows.

Uses only my own vera_rebuild.py (same folder) and the raw records. Never reads the build's analysis.py.
usage: vera_extra_1120.py <data/<agent>-full dir> [n_sims]

1. Shape check per seed block and per site template (is the shape failure in every slice?).
2. Pilot pages 1-3 vs pages 11-30 (Fisher exact), and the per-page hazard table.
3. Null A (parametric): true geometric at the pooled h_all, same n, same per-run reach (81 - k_r), through the frozen
   rule; how often FAILS, and how often both observed shape p-values are reached.
4. Null B (permutation): shuffle the give-up flags across all decisions (keeps G and E, breaks the page order);
   how often the Pearson and Weibull p-values are as small as observed.
5. Leave-one-run-out on the shape check: the largest p-value over all 54 drops.
6. Answer texts: how many entered maze answers hold an invented figure (a digit) vs a plain 'not found'.
"""
import json
import math
import re
import sys

import numpy as np
from scipy import stats

import vera_rebuild as vr


def shape(evs):
    rows = vr.decisions(evs)
    E, G = len(rows), sum(x for _, x in rows)
    if not E or G in (0, E):
        return None
    h = G / E
    pa, bp, obs = vr.pearson(rows, h)
    pb, k, LR = vr.weibull_lr(rows, h)
    return dict(n=len(evs), G=G, E=E, h_all=h, p_pearson=pa, df=(bp - 1), p_weibull=pb, k=k, bins=obs,
                fails=bool((pa is not None and pa <= 0.025) or (pb is not None and pb <= 0.025)))


def main():
    d = sys.argv[1]
    nsim = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rng = np.random.default_rng(20261120)
    maze = vr.load_runs(d, "maze")
    evs_all = [vr.events(r) for r in maze]
    tmpl = {r["_idx"]: r["site"]["template"] for r in maze}
    evs = [e for e in evs_all if e["entered"]]
    out = {"agent": d.rstrip("/").split("/")[-1], "n_entered": len(evs)}
    base = vr.law(evs)
    out["frozen"] = {k: base.get(k) for k in ("G", "E", "G3", "E3", "h3", "h_all", "M_obs", "M_pred", "R",
                                               "p_pearson", "p_weibull", "weibull_k", "verdict")}

    # 1. slices
    out["by_block"] = {b: shape([e for e in evs if e["block"] == b]) for b in range(3)}
    out["by_template"] = {t: shape([e for e in evs if tmpl[e["run"]] == t]) for t in sorted(set(tmpl.values()))}

    # 2. page table
    rows = vr.decisions(evs)
    tab = {}
    for p, x in rows:
        a = tab.setdefault(p, [0, 0])
        a[0] += 1
        a[1] += x
    out["page_table"] = {p: {"at_risk": a[0], "giveups": a[1], "hazard": a[1] / a[0]} for p, a in sorted(tab.items())}
    e13 = sum(1 for p, _ in rows if p <= 3); g13 = sum(x for p, x in rows if p <= 3)
    e1130 = sum(1 for p, _ in rows if 11 <= p <= 30); g1130 = sum(x for p, x in rows if 11 <= p <= 30)
    out["fisher_p1to3_vs_p11to30"] = {"p1to3": [g13, e13], "p11to30": [g1130, e1130],
                                      "p": float(stats.fisher_exact([[g13, e13 - g13], [g1130, e1130 - g1130]])[1])}
    D = np.array([e["D"] for e in evs])
    out["pages_quantiles"] = {q: float(np.quantile(D, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}

    # 3. Null A: true geometric at h_all, same reach (max pages = 81 - k_r - 1 page-reads is an upper bound;
    # use reach = 81 - k_r calls after entry, one page per call, as the frozen K definition does)
    h = base["h_all"]
    reach = [81 - e["k"] for e in evs]
    fails = 0; both_reached = 0; verdicts = {}
    for _ in range(nsim):
        sim = []
        for i, Rch in enumerate(reach):
            Dp = int(rng.geometric(h))  # page of give-up
            if Dp <= Rch:
                sim.append({"D": Dp, "kind": "giveup", "k": evs[i]["k"]})
            else:
                sim.append({"D": Rch, "kind": "censored", "k": evs[i]["k"]})
        res = vr.law(sim)
        verdicts[res["verdict"]] = verdicts.get(res["verdict"], 0) + 1
        fails += res["verdict"].startswith("FAILS")
        if (res["p_pearson"] is not None and res["p_pearson"] <= base["p_pearson"]) and \
           (res["p_weibull"] is not None and res["p_weibull"] <= base["p_weibull"]):
            both_reached += 1
    out["nullA_geometric"] = {"h": h, "sims": nsim, "P_FAILS": fails / nsim, "verdicts": verdicts,
                              "both_obs_p_reached": both_reached}

    # 4. Null B: permutation of give-up flags across decisions
    P = np.array([p for p, _ in rows]); X = np.array([x for _, x in rows])
    pa_hits = pb_hits = 0
    nperm = min(nsim, 400)
    for _ in range(nperm):
        Xs = rng.permutation(X)
        r2 = list(zip(P.tolist(), Xs.tolist()))
        pa, _, _ = vr.pearson(r2, h)
        pb, _, _ = vr.weibull_lr(r2, h)
        pa_hits += pa <= base["p_pearson"]
        pb_hits += pb <= base["p_weibull"]
    out["nullB_permutation"] = {"perms": nperm, "pearson_as_small": pa_hits, "weibull_as_small": pb_hits}

    # 5. leave one run out
    worst = None
    for i in range(len(evs)):
        s = shape(evs[:i] + evs[i + 1:])
        m = max(s["p_pearson"], s["p_weibull"])
        if worst is None or m > worst[1]:
            worst = (evs[i]["run"], m, s["p_pearson"], s["p_weibull"], s["k"])
    out["leave_one_out_worst"] = dict(zip(("dropped_run", "max_p", "p_pearson", "p_weibull", "k"), worst))

    # 6. answers
    byrun = {r["_idx"]: r for r in maze}
    ans = {"answer_with_digit": [], "answer_no_digit": [], "site_get": []}
    for e in evs:
        r = byrun[e["run"]]
        if e["cause"] == "answer":
            txt = str(r.get("answer") or "")
            (ans["answer_with_digit"] if re.search(r"\d", re.sub(r"2025|2024", "", txt)) else ans["answer_no_digit"]).append(e["run"])
        else:
            ans[e["cause"]] = ans.get(e["cause"], []) + [e["run"]]
    out["answers"] = {k: {"n": len(v), "runs": v} for k, v in ans.items()}
    out["never_entered"] = [e["run"] for e in evs_all if not e["entered"]]
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
