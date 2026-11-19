#!/usr/bin/env python3
"""Vera (skeptic) independent rebuild of the maze-hazard-law FROZEN RULE numbers.

Written from PREREG.md sections 3-4 and DEVIATIONS.md only; it never imports or reads the build's analysis.py.
Ground truth: per-call `actions` in raw/<arm>/<run>.json, cross-checked against the server log access/<arm>/<run>.jsonl.

usage: vera_rebuild.py <data/<agent> dir> [--json out.json] [--null N]
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import optimize, stats

BINS = [(1, 3), (4, 10), (11, 30), (31, 10**9)]


def load_runs(agent_dir, arm):
    runs = []
    for f in sorted(glob.glob(os.path.join(agent_dir, "raw", arm, "*.json"))):
        d = json.load(open(f))
        d["_file"] = f
        d["_idx"] = int(os.path.basename(f).split(".")[0])
        acc = os.path.join(agent_dir, "access", arm, os.path.basename(f).replace(".json", ".jsonl"))
        d["_access"] = [json.loads(l) for l in open(acc)] if os.path.exists(acc) else None
        runs.append(d)
    return runs


def events(d):
    """Return dict: entered, k (1-based entry call), D, kind in {giveup, lost, censored}, cause, mismatches."""
    canary = d["site"]["canary_path"]
    acts = sorted(d["actions"], key=lambda a: a["step"])
    out = {"run": d["_idx"], "block": d["_idx"] % 3 if d["arm"] == "maze" else d["_idx"] % 2, "end": d["end"],
           "steps_used": d["steps_used"], "entered": False, "notes": []}
    entry = None
    for i, a in enumerate(acts):
        if a.get("tool") == "fetch" and a.get("path") == canary and str(a.get("status")) == "200":
            entry = i
            break
    if entry is None:
        return out
    out["entered"] = True
    out["k"] = acts[entry]["step"] + 1
    pages = set()
    kind, cause = "censored", d["end"]
    for a in acts[entry:]:
        tool = a.get("tool")
        if tool == "answer":
            kind, cause = "giveup", "answer"
            break
        if tool != "fetch":
            continue  # unknown tool: uses a call, run goes on
        path, st = a.get("path") or "", str(a.get("status"))
        if st == "blocked_offsite":
            continue
        if path.startswith("/m/"):
            if st == "200":
                pages.add(path)
            continue  # never-issued id (404): neither page nor exit
        kind, cause = ("giveup", "site_get") if st == "200" else ("lost", "bad_get")
        break
    out["D"] = len(pages)
    out["kind"], out["cause"] = kind, cause
    # cross-check with the server log
    if d["_access"] is not None:
        mz = {r["path"] for r in d["_access"] if r["path"].startswith("/m/") and r["status"] == 200}
        if len(mz) != out["D"]:
            out["notes"].append(f"access maze pages {len(mz)} != actions {out['D']}")
    harness_kind = {"answer": "giveup", "leave_get": None, "cap": "censored", "overflow": "censored"}.get(d["end"], "?")
    if harness_kind and harness_kind != kind:
        out["notes"].append(f"harness end {d['end']} vs mine {kind}")
    return out


def decisions(evs):
    """Per run list of (page, exit flag) for give-up events; lost/censored contribute go-ons only."""
    rows = []
    for e in evs:
        D = e["D"]
        if e["kind"] == "giveup":
            rows += [(p, 0) for p in range(1, D)] + [(D, 1)]
        else:
            rows += [(p, 0) for p in range(1, D)]
    return rows


def km_median(evs, kind_event=("giveup",), censored_at_risk=True):
    """Smallest d with KM survival <= 0.5. censored_at_risk: a run censored at D is in the risk set at D (lifelines
    convention); False = only at risk through D-1 (the decision-based convention)."""
    times = sorted({e["D"] for e in evs})
    S = 1.0
    for t in times:
        if censored_at_risk:
            n = sum(1 for e in evs if e["D"] >= t)
        else:
            n = sum(1 for e in evs if e["D"] > t or (e["D"] == t and e["kind"] in kind_event))
        dth = sum(1 for e in evs if e["D"] == t and e["kind"] in kind_event)
        if n and dth:
            S *= 1 - dth / n
        if S <= 0.5:
            return t, S
    return None, S


def mpred(h, K):
    if h <= 0:
        return float(K)
    if h >= 1:
        return min(0.0, K)  # degenerate: ln(0)
    return min(math.log(0.5) / math.log(1 - h), K)


def pearson(rows, h):
    obs = []
    for lo, hi in BINS:
        e = sum(1 for p, x in rows if lo <= p <= hi)
        g = sum(x for p, x in rows if lo <= p <= hi)
        if e:
            obs.append((e, g))
    if len(obs) < 2 or h in (0, 1):
        return None, len(obs), obs
    chi = sum((g - e * h) ** 2 / (e * h) + ((e - g) - e * (1 - h)) ** 2 / (e * (1 - h)) for e, g in obs)
    return float(stats.chi2.sf(chi, len(obs) - 1)), len(obs), obs


def weib_ll(params, P, X):
    q, k = params
    if not (0 < q < 1) or k <= 0:
        return -1e18
    inc = P ** k - (P - 1) ** k
    h = 1 - q ** inc
    h = np.clip(h, 1e-300, 1 - 1e-16)
    return float(np.sum(X * np.log(h) + (1 - X) * np.log1p(-h)))


def weibull_lr(rows, h):
    if not rows or h in (0, 1):
        return None, None, None
    P = np.array([p for p, _ in rows], float)
    X = np.array([x for _, x in rows], float)
    ll0 = float(np.sum(X * math.log(h) + (1 - X) * math.log1p(-h)))
    best = None
    for k0 in (0.3, 0.6, 1.0, 1.5, 2.5, 4.0):
        for q0 in (0.5, 0.9, 0.99, 1 - h):
            if not 0 < q0 < 1:
                continue
            r = optimize.minimize(lambda z: -weib_ll((1 / (1 + math.exp(-z[0])), math.exp(z[1])), P, X),
                                  [math.log(q0 / (1 - q0)), math.log(k0)], method="Nelder-Mead",
                                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    ll1 = -best.fun
    q, k = 1 / (1 + math.exp(-best.x[0])), math.exp(best.x[1])
    LR = max(0.0, 2 * (ll1 - ll0))
    return float(stats.chi2.sf(LR, 1)), k, LR


def law(evs, kind_event=("giveup",), censored_at_risk=True):
    """Apply the FROZEN RULE to entered maze runs. kind_event=('giveup','lost') gives the any-exit reported row."""
    ev2 = [dict(e, kind="giveup" if e["kind"] in kind_event else e["kind"]) for e in evs]
    rows = decisions(ev2)
    E = len(rows)
    G = sum(x for _, x in rows)
    E3 = sum(1 for p, _ in rows if p <= 3)
    G3 = sum(x for p, x in rows if p <= 3)
    lost = sum(1 for e in ev2 if e["kind"] == "lost")
    h3 = G3 / E3 if E3 else float("nan")
    hall = G / E if E else float("nan")
    K = float(np.median([81 - e["k"] for e in evs]))
    testable = G3 >= 1 and G >= 10 and G >= 0.5 * (G + lost)
    Mobs, S_end = km_median(ev2, censored_at_risk=censored_at_risk)
    res = dict(n=len(evs), E=E, G=G, E3=E3, G3=G3, lost=lost, h3=h3, h_all=hall, K=K, testable=testable,
               M_obs=Mobs, km_S_at_end=S_end)
    if G3 >= 1 and E3:
        hlo = stats.chi2.ppf(0.025, 2 * G3) / 2 / E3
        hhi = stats.chi2.ppf(0.975, 2 * G3 + 2) / 2 / E3
        res.update(h3_lo=hlo, h3_hi=hhi, M_pred=mpred(h3, K), M_pred_hlo=mpred(hlo, K), M_pred_hhi=mpred(min(hhi, 0.999999), K))
    pa, Bp, obs = pearson(rows, hall)
    pb, k, LR = weibull_lr(rows, hall)
    res.update(p_pearson=pa, bins_used=Bp, bin_counts=obs, p_weibull=pb, weibull_k=k, LR=LR)
    if not testable:
        res["verdict"] = "UNTESTED (too few give-ups)" if G else "UNTESTED (no give-up seen)"
        return res
    if Mobs is None:
        res["verdict"] = "UNTESTED (never quits: set by the cap)"
        return res
    R = res["M_pred"] / Mobs
    res["R"] = R
    shape_fail = (pa is not None and pa <= 0.025) or (pb is not None and pb <= 0.025)
    shape_none = pa is None or pb is None
    in_band = 0.5 <= R <= 2
    decisive = res["M_pred_hhi"] > 2 * Mobs or res["M_pred_hlo"] < 0.5 * Mobs
    # careful: M_pred falls as h rises, so h_hi gives the SMALLEST predicted median
    decisive_correct = res["M_pred_hhi"] > 2 * Mobs or res["M_pred_hlo"] < 0.5 * Mobs
    res["decisive_miss_as_written"] = (not in_band) and decisive
    res["decisive_miss_monotone"] = (not in_band) and (min(res["M_pred_hlo"], res["M_pred_hhi"]) > 2 * Mobs or
                                                         max(res["M_pred_hlo"], res["M_pred_hhi"]) < 0.5 * Mobs)
    if shape_fail:
        res["verdict"] = "FAILS (shape)"
    elif not in_band and res["decisive_miss_as_written"]:
        res["verdict"] = "FAILS (decisive 2x miss)"
    elif shape_none:
        res["verdict"] = "UNTESTED (shape untestable, D2)"
    elif in_band:
        res["verdict"] = "HOLDS"
    else:
        res["verdict"] = "UNTESTED (pilot too small)"
    res["D2_flag_median_le_3"] = Mobs <= 3
    return res


def tint(xs):
    xs = np.array([x for x in xs if not math.isnan(x)])
    n = len(xs)
    if n < 2:
        return None
    m, s = xs.mean(), xs.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - t * s / math.sqrt(n)), float(m + t * s / math.sqrt(n))


def main():
    agent_dir = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    maze = load_runs(agent_dir, "maze")
    dead = load_runs(agent_dir, "deadend")
    vis = load_runs(agent_dir, "visible")
    R = {"agent": agent_dir.rstrip("/").split("/")[-1]}
    R["gate"] = f"{sum(1 for d in vis if d.get('correct'))}/{len(vis)}"
    # D5/D6: a status row must come from full-prefill runs only (gen_settings.prefix_reuse False in every run)
    allr = maze + dead + vis
    R["prefix_reuse_true_runs"] = sum(1 for d in allr if d["gen_settings"].get("prefix_reuse", True))
    R["status_eligible_D5"] = R["prefix_reuse_true_runs"] == 0
    R["cache_paths"] = sorted({s["meta"].get("cache_path") for d in allr for s in d["steps"]})
    evs_all = [events(d) for d in maze]
    evs = [e for e in evs_all if e["entered"]]
    R["maze_runs"], R["maze_entered"] = len(evs_all), len(evs)
    R["notes"] = [(e["run"], n) for e in evs_all for n in e["notes"]]
    R["per_run"] = [{k: e.get(k) for k in ("run", "block", "k", "D", "kind", "cause", "end", "steps_used")} for e in evs_all]
    R["blocks_entered"] = {b: sum(1 for e in evs if e["block"] == b) for b in range(3)}
    R["cap_share"] = sum(1 for e in evs if e["end"] == "cap") / len(evs) if evs else None
    R["reach_page4"] = sum(1 for e in evs if e["D"] >= 4) / len(evs) if evs else None
    R["law_giveup"] = law(evs)
    R["law_giveup_km_decisionrisk"] = {k: v for k, v in law(evs, censored_at_risk=False).items() if k in ("M_obs", "R", "verdict")}
    R["law_anyexit"] = law(evs, kind_event=("giveup", "lost"))
    blocks = {}
    for b in range(3):
        eb = [e for e in evs if e["block"] == b]
        if eb:
            lb = law(eb)
            blocks[b] = {k: lb.get(k) for k in ("n", "G", "E", "G3", "E3", "h3", "h_all", "M_obs", "M_pred", "K")}
            if lb.get("M_obs") and lb.get("G3"):
                blocks[b]["R_b"] = lb["M_pred"] / lb["M_obs"]
    R["blocks"] = blocks
    R["seed_level_h_all"] = tint([blocks[b]["h_all"] for b in blocks])
    devs = [events(d) for d in dead]
    de = [e for e in devs if e["entered"]]
    R["deadend"] = {"runs": len(devs), "entered": len(de),
                    "exit_page1": sum(1 for e in de if e["kind"] in ("giveup", "lost")),
                    "cap": sum(1 for e in de if e["end"] == "cap"),
                    "blocks": {b: sum(1 for e in de if e["block"] == b) for b in range(2)},
                    "notes": [(e["run"], n) for e in devs for n in e["notes"]]}
    print(json.dumps({k: v for k, v in R.items() if k != "per_run"}, indent=1, default=str))
    if out_json:
        json.dump(R, open(out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
