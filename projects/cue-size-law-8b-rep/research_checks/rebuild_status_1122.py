"""Independent stdlib rebuild of the frozen status (Bruno, theory-1122-cue-size-law-8b continuation).

Does NOT import analyze_8b.py. Recomputes every eval's gap from the per-prompt P(X) arrays (p_off, p_on), D(t) =
gap(t) - gap(0) with gap(0) from the step-0 eval, the dense switch time T (first eval with D >= 0.10 and D(next) >=
0.10), the per-seed OLS slope of ln T on ln share over sizes with |gap0| < 0.02 (read from data/gates.json), the
seed-level t-interval (df 2 quantile in closed form), C1 (full B, never switched), and compares the stored D/T.
Also compares the partial mini copy of C1 seed 6 (killed at the 2026-11-20 wrapup at update 2624) with the laptop
copy, point by point. Run: python3 research_checks/rebuild_status_1122.py  (from the project dir; stdlib only).
"""
import ast
import glob
import json
import math
import os
import sys
from fractions import Fraction

THRESH = 0.10
B = 8192
BAND = (Fraction(-14, 10), Fraction(-7, 10))


def mean(xs):
    return math.fsum(xs) / len(xs)


def curve(run):
    """Merge dev_curve and dense_curve on step; gap recomputed from per-prompt arrays when present."""
    pts = {}
    for src in ("dev_curve", "dense_curve"):
        for p in run.get(src) or []:
            s = int(p["step"])
            if s in pts:
                continue
            if "p_off" in p and "p_on" in p:
                g = mean(p["p_off"]) - mean(p["p_on"])
            else:
                g = float(p["px_off"]) - float(p["px_on"])
            pts[s] = (g, p.get("D"))
    return pts


def switch(run, gap0_override=None):
    pts = curve(run)
    g0 = pts[0][0] if 0 in pts else run["gap0"]
    c = sorted((s, g - g0, d) for s, (g, d) in pts.items() if s > 0)
    max_dd = max((abs(d - dd) for _, d, dd in c if dd is not None), default=0.0)
    maxD = max(d for _, d, _ in c)
    for i in range(len(c) - 1):
        if c[i][1] >= THRESH and c[i + 1][1] >= THRESH:
            return c[i][0], False, max_dd, maxD, g0
    return int(run.get("budget", B)), True, max_dd, maxD, g0


def ols(xs, ys):
    mx, my = mean(xs), mean(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    return math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def t975(df):
    if df == 1:  # Student t df 1 = Cauchy: F(t) = 1/2 + atan(t)/pi  =>  t = tan(pi (0.975 - 1/2))
        return math.tan(math.pi * 0.475)
    if df != 2:
        raise SystemExit("closed form only for df 1 and 2")
    q = 0.95  # Student t df 2: F(t) = 1/2 + t / (2 sqrt(2 + t^2))  =>  t = q sqrt(2 / (1 - q^2)), q = 2F - 1
    return q * math.sqrt(2 / (1 - q * q))


def main():
    gates = json.load(open("data/gates.json"))
    gap0_gate = {int(k): v for k, v in gates["gap0"].items()} if isinstance(gates["gap0"], dict) else dict(
        enumerate(gates["gap0"]))
    in_fit = {k: abs(v) < 0.02 for k, v in gap0_gate.items()}
    out = {"gates": {"G1": gates["G1"]["pass"], "G2a": gates["G2"]["a_pass"], "G2b": gates["G2"]["b_pass"],
                     "gap0_by_k": gap0_gate, "in_fit": in_fit}}
    runs = {}
    for f in sorted(glob.glob("data/runs/*.json")):
        r = json.load(open(f))
        a = r["args"]
        T, cens, max_dd, maxD, g0 = switch(r)
        runs[r["name"]] = dict(arm=a["arm"], seed=int(a["seed"]), k=r.get("k"), share=r.get("share"),
                               machine=r.get("machine"), steps_run=r.get("steps_run"), budget=r.get("budget"),
                               alpha=a.get("cue_alpha"), T=T, cens=cens, stored_T=r.get("T_dense"),
                               max_abs_D_minus_stored=max_dd, maxD=maxD, gap0=g0)
    out["runs"] = runs
    main_runs = [v for v in runs.values() if v["arm"] == "cue" and (v["alpha"] or 0) > 0]
    slopes = {}
    for s in sorted({v["seed"] for v in main_runs}):
        rs = [v for v in main_runs if v["seed"] == s and in_fit.get(v["k"], False) and not v["cens"]]
        ks = sorted(v["k"] for v in main_runs if v["seed"] == s)
        if ks == list(range(6)) and len(rs) >= 4:
            slopes[s] = ols([math.log(v["share"]) for v in rs], [math.log(v["T"]) for v in rs])
    n = len(slopes)
    m = mean(list(slopes.values()))
    sd = math.sqrt(math.fsum((b - m) ** 2 for b in slopes.values()) / (n - 1))
    h = t975(n - 1) * sd / math.sqrt(n)
    lo, hi = m - h, m + h
    c1 = {k: v for k, v in runs.items() if v["arm"] == "shuf"}
    c1_ok = len({v["seed"] for v in c1.values()}) >= 2 and all(
        v["cens"] and v["steps_run"] >= v["budget"] for v in c1.values())
    inside = BAND[0] < Fraction(lo) and Fraction(hi) < BAND[1]
    gates_ok = all(out["gates"][g] for g in ("G1", "G2a", "G2b"))
    status = "CONFIRMED" if (n >= 3 and inside and c1_ok and gates_ok) else "NOT CONFIRMED"
    out.update(slopes=slopes, seed_level=dict(n=n, mean=m, sd=sd, t975=t975(n - 1), ci=[lo, hi]),
               c1=dict(ok=c1_ok, runs={k: dict(seed=v["seed"], machine=v["machine"], steps_run=v["steps_run"],
                                                 maxD=v["maxD"], switched=not v["cens"]) for k, v in c1.items()}),
               n_censored_main=sum(v["cens"] for v in main_runs), status=status)

    # cross-machine determinism: a partial log on one machine vs the finished run JSON from the other machine
    def log_vs_json(log, js):
        if not (os.path.exists(log) and os.path.exists(js)):
            return None
        other = curve(json.load(open(js)))
        diffs = []
        for line in open(log):
            if line.startswith("step "):
                rest = line.split(" ", 2)[2]
                p = ast.literal_eval(rest[:rest.index("}") + 1])
                s = int(p["step"])
                if s in other:
                    diffs.append((s, abs(p["gap"] - other[s][0])))
        return dict(n_points=len(diffs), n_equal=sum(d == 0.0 for _, d in diffs),
                    max_abs_gap_diff=max(d for _, d in diffs), last_step=diffs[-1][0])

    # C1 seed 6: mini M2 (killed by the oracle watch at the 11-20 wrapup) vs the laptop copy
    out["c1_s6_mini_vs_laptop"] = log_vs_json("data/mini/M2_partial/m8_c1_s6.log", "data/runs/l8_c1_s6.json")
    # S9 u1127 k5 s7: laptop R3 partial (failed job 5f3rwdo6um, ends at update 576) vs the mini M3 copy
    out["s9_u1127_k5_s7_laptop_vs_mini"] = log_vs_json("data/laptop/R3/l8u1127_k5_s7.log",
                                                       "data/runs_explore/m8u1127_k5_s7.json")
    # C2 pipeline checks (cue 0): pass line max |gap| = 0 over every eval
    out["c2"] = {k: dict(seed=v["seed"], steps_run=v["steps_run"],
                         max_abs_gap=max(abs(g) for g, _ in curve(json.load(open(f"data/runs/{k}.json"))).values()))
                 for k, v in runs.items() if v["arm"] == "cue" and not (v["alpha"] or 0) > 0}
    # S9 fresh directions (exploratory): same slope rule, gap0 exclusion from gates.json s9[seed].gap0
    s9 = {}
    for s, g in gates.get("s9", {}).items():
        fs = sorted(glob.glob(f"data/runs_explore/*u{g['u_file'][-8:-4]}_k*_s{s}.json"))
        rr = []
        for f in fs:
            r = json.load(open(f))
            T, cens, _, _, _ = switch(r)
            rr.append(dict(name=r["name"], k=r["k"], share=r["share"], T=T, cens=cens, machine=r["machine"],
                           in_fit=abs(g["gap0"][str(r["k"])]) < 0.02))
        use = [x for x in rr if x["in_fit"] and not x["cens"]]
        fin = sorted(x["k"] for x in rr) == list(range(6))
        s9[s] = dict(runs=rr, finished=fin, slope=(ols([math.log(x["share"]) for x in use],
                                                        [math.log(x["T"]) for x in use]) if fin and len(use) >= 4 else None))
    bs = [v["slope"] for v in s9.values() if v["slope"] is not None]
    if len(bs) >= 2:
        mm = mean(bs)
        ss = math.sqrt(math.fsum((b - mm) ** 2 for b in bs) / (len(bs) - 1))
        hh = t975(len(bs) - 1) * ss / math.sqrt(len(bs))
        out["s9"] = dict(per_seed=s9, n=len(bs), mean=mm, sd=ss, ci=[mm - hh, mm + hh])
    else:
        out["s9"] = dict(per_seed=s9, n=len(bs))
    json.dump(out, open("research_checks/rebuild_status_1122.json", "w"), indent=1, default=str)
    print("slopes", {k: round(v, 6) for k, v in slopes.items()})
    print("seed-level", round(m, 6), [round(lo, 6), round(hi, 6)], "t", round(t975(n - 1), 6))
    print("C1", out["c1"])
    print("max |D recomputed - D stored| over all runs", max(v["max_abs_D_minus_stored"] for v in runs.values()))
    print("T mismatches vs stored", [k for k, v in runs.items() if v["stored_T"] is not None and v["stored_T"] != v["T"]])
    print("main censored", out["n_censored_main"], "gates", out["gates"]["G1"], out["gates"]["G2a"], out["gates"]["G2b"])
    print("mini vs laptop C1 s6", out.get("c1_s6_mini_vs_laptop"))
    print("laptop vs mini S9 u1127 k5 s7", out.get("s9_u1127_k5_s7_laptop_vs_mini"))
    print("C2", out["c2"])
    print("S9", {s: (v["slope"], [(x["k"], x["T"], x["machine"], x["in_fit"]) for x in v["runs"]])
                 for s, v in out["s9"]["per_seed"].items()},
          {k: v for k, v in out["s9"].items() if k != "per_seed"})
    print("STATUS", status)


if __name__ == "__main__":
    sys.exit(main())
