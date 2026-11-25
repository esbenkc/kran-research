"""Apply PREREG.md 2(6) mechanically, plus the secondaries of 2(7). Reads predictions.json (pushed before any held-out
cell ran) and the raw cell files; writes results.json.

    .venv/bin/python score.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE / "data"
LAWS = ["L", "L_z", "N1", "N2", "N3", "N4"]
BAR = 5.0
SEEN_PEEK = "24x24x12r3"   # Bruno's disclosed smoke peek (DEVIATIONS.md): sensitivity without this shape


def skey(r):
    return f"nb{r['nbeta']:g}_q{r['q']:g}_c{r['c']:g}"


def load_cells(arm):
    out = {}
    d = DATA / f"cells_{arm}"
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        for r in j["rows"]:
            out.setdefault(f"{r['case']}|{skey(r)}", []).append(r)
    return out


def load_gibbs(arm):
    out = {}
    d = DATA / f"gibbs_{arm}"
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        for r in json.loads(f.read_text())["rows"]:
            out.setdefault(f"{r['case']}|nb{r['nbeta']:g}_q{r['q']:g}", []).append(r)
    return out


def measure(rows):
    """Seed-mean e and the convergence verdict (PREREG 2(5): halves within 2 points in every seed; NaN/diverged)."""
    ok = all(r["alive"] and r["halves_ok"] and math.isfinite(r["e"]) for r in rows)
    es = [r["e"] for r in rows]
    return {"e_meas": float(np.mean(es)) if all(math.isfinite(e) for e in es) else float("nan"),
            "e_seeds": {str(r["seed"]): r["e"] for r in rows}, "converged": ok, "n_seeds": len(rows)}


def score(cells, law):
    miss = []
    for c in cells:
        if not c["converged"] or not math.isfinite(c["e_meas"]):
            miss.append(100.0)
        else:
            miss.append(100 * abs(c["pred"][law] - c["e_meas"]))
    miss = np.array(miss)
    return {"n": int(len(miss)), "share_within_5": float((miss <= BAR).mean()) if len(miss) else None,
            "median_miss": float(np.median(miss)) if len(miss) else None,
            "max_miss": float(miss.max()) if len(miss) else None, "misses": miss.tolist()}


def seed_level(cells, law, seeds):
    shares = []
    for sd in seeds:
        m = []
        for c in cells:
            e = c["e_seeds"].get(str(sd))
            if e is None or not c["converged"] or not math.isfinite(e):
                m.append(100.0)
            else:
                m.append(100 * abs(c["pred"][law] - e))
        shares.append(float((np.array(m) <= BAR).mean()))
    n = len(shares)
    mu = float(np.mean(shares))
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * np.std(shares, ddof=1) / math.sqrt(n))
    else:
        half = float("nan")
    return {"per_seed": shares, "mean": mu, "ci95": [mu - half, mu + half], "method": f"t-interval over {n} seeds"}


def status(prim):
    sL = score(prim, "L")
    nulls = {n: score(prim, n) for n in ("N1", "N2", "N3", "N4")}
    spread = 100 * (max(c["e_meas"] for c in prim if math.isfinite(c["e_meas"]))
                    - min(c["e_meas"] for c in prim if math.isfinite(c["e_meas"])))
    best_null = min(v["median_miss"] for v in nulls.values())
    skill = all(sL["median_miss"] <= 0.5 * v["median_miss"] for v in nulls.values())
    if spread < 20:
        st = "UNCONFIRMED"
        why = f"spread {spread:.1f} < 20 points"
    elif sL["share_within_5"] >= 0.80 and skill:
        st, why = "CONFIRMED", "share >= 0.80 and median miss <= half of every null"
    elif sL["share_within_5"] < 0.60 or sL["median_miss"] >= best_null:
        st = "REJECTED"
        why = "share < 0.60" if sL["share_within_5"] < 0.60 else "median miss >= best null"
    else:
        st, why = "UNCONFIRMED", "between the bars"
    return {"status": st, "why": why, "spread_pts": spread, "L": {k: v for k, v in sL.items() if k != "misses"},
            "nulls": {k: {kk: vv for kk, vv in v.items() if kk != "misses"} for k, v in nulls.items()},
            "skill_vs_all_nulls": skill, "best_null_median": best_null}


def main():
    P = json.loads((HERE / "predictions.json").read_text())
    pred = P["cells"]
    S = load_cells("S")
    Pm = load_cells("P")
    G = load_gibbs("S")
    GP = load_gibbs("P")
    cells = []
    for k, p in pred.items():
        if k not in S:
            continue
        m = measure(S[k])
        g = G.get(f"{p['case']}|nb{p['nbeta']:g}_q{p['q']:g}", [])
        gp = GP.get(f"{p['case']}|nb{p['nbeta']:g}_q{p['q']:g}", [])
        mP = measure(Pm[k]) if k in Pm else None
        cells.append({"key": k, "case": p["case"], "group": p["group"], "primary": p["primary"],
                      "nbeta": p["nbeta"], "q": p["q"], "c": p["c"], "gamma": p["gamma"], "T": p["T"], **m,
                      "pred": {law: p["e_" + law.replace("_", "")] for law in LAWS},
                      "e_gibbs": float(np.mean([r["e_gibbs"] for r in g])) if g else None,
                      "e_gibbs_P": float(np.mean([r["e_gibbs"] for r in gp])) if gp else None,
                      "e_meas_P": None if mP is None else mP["e_meas"],
                      "converged_P": None if mP is None else mP["converged"]})
    prim = [c for c in cells if c["primary"]]
    res = {"prereg_commit": P["prereg_commit"], "n_cells_scored": len(cells), "n_primary": len(prim)}
    if prim:
        res["primary"] = status(prim)
        res["primary"]["seed_level"] = seed_level(prim, "L", [0, 1, 2])
        res["primary"]["n_not_converged"] = sum(not c["converged"] for c in prim)
        res["primary"]["n_expected"] = 180
        nopeek = [c for c in prim if c["case"] != SEEN_PEEK]
        res["sensitivity_without_24x24x12r3"] = status(nopeek)
        res["secondary"] = {
            "L_z_primary": {k: v for k, v in score(prim, "L_z").items() if k != "misses"},
            "L_z_blind_all_q": {k: v for k, v in score([c for c in cells if c["group"] == "blind"], "L_z").items()
                                if k != "misses"},
            "S1_N3_share_primary": score(prim, "N3")["share_within_5"],
            "S4_strong_spring_blind": {k: v for k, v in score([c for c in cells if c["group"] == "blind"
                                                                and c["q"] < 0.1], "L").items() if k != "misses"},
            "S6_seen": {k: v for k, v in score([c for c in cells if c["group"] == "seen" and c["q"] >= 0.1],
                                               "L").items() if k != "misses"},
            "control_16x16x8r8": {k: v for k, v in score([c for c in cells if c["group"] == "control"
                                                         and c["q"] >= 0.1], "L").items() if k != "misses"},
        }
        conv = [c for c in prim if c["converged"]]
        res["secondary"]["diagnostic_converged_only"] = {"n": len(conv), **{law: {k: v for k, v in score(conv, law).items()
                                                         if k != "misses"} for law in LAWS}}
        gc = [{**c, "e_meas": c["e_gibbs"], "converged": True} for c in prim if c["e_gibbs"] is not None]
        res["secondary"]["law_vs_gibbs"] = {k: v for k, v in score(gc, "L").items() if k != "misses"}
        primP = [c for c in prim if c["e_meas_P"] is not None]
        if primP:
            pc = [{**c, "e_meas": c["e_meas_P"], "converged": c["converged_P"]} for c in primP]
            res["secondary"]["S3_armP_primary"] = {k: v for k, v in score(pc, "L").items() if k != "misses"}
        # error split per primary cell: target bias (Gibbs - truth) and sampler bias (SGLD - Gibbs)
        gs = [c for c in prim if c["e_gibbs"] is not None and math.isfinite(c["e_meas"])]
        if gs:
            res["secondary"]["error_split"] = {
                "n": len(gs), "median_abs_target_pts": float(np.median([100 * abs(c["e_gibbs"]) for c in gs])),
                "median_abs_sampler_pts": float(np.median([100 * abs(c["e_meas"] - c["e_gibbs"]) for c in gs])),
                "max_abs_sampler_pts": float(np.max([100 * abs(c["e_meas"] - c["e_gibbs"]) for c in gs]))}
    res["cells"] = cells
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    if prim:
        p = res["primary"]
        print(f"STATUS {p['status']} ({p['why']}); L share {p['L']['share_within_5']:.3f} median "
              f"{p['L']['median_miss']:.2f} max {p['L']['max_miss']:.1f}; spread {p['spread_pts']:.1f}; "
              f"nulls " + ", ".join(f"{k} {v['median_miss']:.2f}/{v['share_within_5']:.2f}" for k, v in p['nulls'].items())
              + f"; seed-level {p['seed_level']['mean']:.3f} {p['seed_level']['ci95']}; not converged "
              f"{p['n_not_converged']}/{len(prim)}")
        s = res["sensitivity_without_24x24x12r3"]
        print(f"without 24x24x12r3: {s['status']} share {s['L']['share_within_5']:.3f}")
        print(json.dumps({k: v for k, v in res["secondary"].items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
