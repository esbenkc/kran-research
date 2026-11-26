"""Vera's independent rebuild of second-tell-8b from the saved per-prompt dev probabilities.

Written from PREREG.md (commit 7b19205af, frozen block lines 68-139 + S1-S7) only; I did not read tt.py or the
analysis script. Inputs: the run JSONs (one per tail) with per-check per-prompt dev P(X) lists.

For each stage-2 tail:
  gap(t)   = mean_dev P(X | no cue) - mean_dev P(X | cue B on)
  gap_base = same for the untrained tail (LoRA off, parent b) on that seed's u_B  (from p_base_off / p_base_on)
  D(t)     = gap(t) - gap_base
  T_B      = first check t (stage-2 updates) with D >= 0.10 at t and at the next check; none by the cap -> censored, 1536.
Status (frozen): R_s,r = max(T_MAIN,s, 8) / max(T_r,s, 8), paired by seed; 95% t-interval of mean ln R (df n-1).
  CONFIRMED: upper < ln 0.5 for every r in {fresh, shuf, coin}; REJECTED: lower > ln 0.5 for shuf or fresh; else UNCONFIRMED.
  Censoring: > 1 censored tail in MAIN or in r -> r cannot give CONFIRMED/REJECTED. Minimum 3 seeds. Gates G1, G2, port.
Usage: python3 vera_check.py <stage2_dir> [<stage1_dir>] [--json out.json]
"""
import json, math, sys, glob, os
import numpy as np
from scipy import stats

CAP = 1536
THR = 0.10
LOGIT_THR = 0.486   # S6: logit(0.3435) - logit(0.2435)
L5 = math.log(0.5)
REFS = ("fresh", "shuf", "coin")


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        if not isinstance(j, dict) or "arm" not in j or "seed" not in j:
            continue
        runs[(j["arm"], int(j["seed"]))] = (f, j)
    return runs


def cue_key(entry):
    # stage 2 logs cue B (p_B) and, every 64 updates, cue A retention (p_A); stage 1 logs p_A only
    if isinstance(entry.get("p_B"), list):
        return "p_B"
    ks = [k for k in entry if k.startswith("p_") and k != "p_off" and isinstance(entry[k], list)]
    assert len(ks) == 1, f"ambiguous cue-on key {ks}"
    return ks[0]


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def switch_time(curve, base, thr, kind="prob"):
    pts = []
    for e in sorted(curve, key=lambda e: e["t"]):
        k = cue_key(e)
        off, on = float(np.mean(e["p_off"])), float(np.mean(e[k]))
        g = (off - on) if kind == "prob" else (logit(off) - logit(on))
        pts.append((int(e["t"]), g - base))
    for i in range(len(pts) - 1):
        if pts[i][1] >= thr and pts[i + 1][1] >= thr:
            return pts[i][0], False, pts
    return CAP, True, pts


def tint(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def rebuild(stage2_dir, stage1_dir=None):
    runs = load(stage2_dir)
    out = {"tails": {}, "notes": []}
    T, cens, D0, px0, Tlog = {}, {}, {}, {}, {}
    for (arm, s), (f, j) in sorted(runs.items()):
        gb_prob = float(np.mean(j["p_base_off"]) - np.mean(j["p_base_on"]))
        gb_logit = logit(float(np.mean(j["p_base_off"]))) - logit(float(np.mean(j["p_base_on"])))
        t, c, pts = switch_time(j["curve"], gb_prob, THR)
        tl, cl, _ = switch_time(j["curve"], gb_logit, LOGIT_THR, "logit")
        e0 = min(j["curve"], key=lambda e: e["t"])
        T[(arm, s)], cens[(arm, s)], Tlog[(arm, s)] = t, c, (tl, cl)
        D0[(arm, s)] = pts[0][1]
        px0[(arm, s)] = float(np.mean(e0["p_off"]))
        stored = j.get("T_B", j.get("T"))
        out["tails"][f"{arm}_s{s}"] = {
            "T_vera": t, "censored": c, "T_stored": stored, "match": (stored == t) or (stored is None and c),
            "gap_base_vera": gb_prob, "gap_base_stored": j.get("gap_base"), "D0": pts[0][1],
            "px_off_t0": px0[(arm, s)], "T_logit": tl, "logit_censored": cl, "last_t": pts[-1][0],
            "b": j.get("b"), "file": os.path.basename(f)}
    seeds = sorted({s for (a, s) in T if a == "main"})
    # stage-1 drop rule
    dropped = []
    if stage1_dir:
        s1 = load(stage1_dir)
        for s in seeds:
            if ("main", s) in s1 and s1[("main", s)][1].get("T_A") is None:
                dropped.append(s)
        out["stage1_TA"] = {f"{a}_s{s}": j.get("T_A") for (a, s), (_, j) in s1.items()}
    kept = [s for s in seeds if s not in dropped]
    out["main_seeds_kept"], out["main_dropped"] = kept, dropped
    lm = lambda a, s: math.log(max(T[(a, s)], 8))
    res = {}
    for r in REFS:
        pair = [s for s in kept if (r, s) in T]
        x = [lm("main", s) - lm(r, s) for s in pair]
        m, lo, hi = tint(x)
        ncm = sum(cens[("main", s)] for s in kept)
        ncr = sum(cens[(r, s)] for s in pair)
        res[r] = {"seeds": pair, "lnR": x, "mean_lnR": m, "lo": lo, "hi": hi, "R": math.exp(m),
                  "R_lo": math.exp(lo), "R_hi": math.exp(hi), "n_cens_main": ncm, "n_cens_ref": ncr,
                  "usable": ncm <= 1 and ncr <= 1 and len(pair) >= 3,
                  "geomean_T": float(np.exp(np.mean([math.log(max(T[(r, s)], 8)) for s in pair]))) if pair else None}
    out["ratios"] = res
    conf = all(res[r]["usable"] and res[r]["hi"] < L5 for r in REFS)
    rej = any(res[r]["usable"] and res[r]["lo"] > L5 for r in ("shuf", "fresh"))
    fresh_T = [T[("fresh", s)] for s in seeds if ("fresh", s) in T]
    port = float(np.median(fresh_T)) if fresh_T else None
    out["port_gate"] = {"fresh_T": fresh_T, "median": port, "pass": port is not None and 266 <= port <= 326}
    status = "CONFIRMED" if conf else ("REJECTED" if rej else "UNCONFIRMED")
    if len(kept) < 3:
        status = "UNCONFIRMED"
    if not out["port_gate"]["pass"] and status != "UNCONFIRMED":
        out["notes"].append(f"port gate fails -> {status} capped to UNCONFIRMED")
        status = "UNCONFIRMED"
    out["status_before_G1G2"] = status
    # prereg outcome: MAIN seeds 5,6,7 each <= 0.5 x geomean T of SHUF and of COIN
    gm = {r: res[r]["geomean_T"] for r in ("shuf", "coin")}
    po = {s: T[("main", s)] <= 0.5 * gm["shuf"] and T[("main", s)] <= 0.5 * gm["coin"] for s in (5, 6, 7) if ("main", s) in T}
    out["prereg_3of3"] = po
    out["prereg_5of5"] = {s: T[("main", s)] <= 0.5 * gm["shuf"] and T[("main", s)] <= 0.5 * gm["coin"] for s in kept}
    # S1 law read vs FRESH
    f = res["fresh"]
    out["S1_law"] = {"R": f["R"], "lo": f["R_lo"], "hi": f["R_hi"],
                     "read": "held" if (f["R_lo"] >= 0.40 and f["R_hi"] <= 0.66) else
                     ("failed" if (f["R_hi"] < 0.40 or f["R_lo"] > 0.66) else "unsettled")}
    # S2 output-path-only
    out["S2"] = {r: res["fresh"]["geomean_T"] / res[r]["geomean_T"] for r in ("shuf", "coin")}
    # S3/S5 per arm
    arms = sorted({a for (a, s) in T})
    out["S3_D0"] = {a: [D0[(a, s)] for s in seeds if (a, s) in D0] for a in arms}
    out["S5_pxoff_t0"] = {a: [px0[(a, s)] for s in seeds if (a, s) in px0] for a in arms}
    # S6 logit units: same ratios
    s6 = {}
    for r in REFS:
        pair = [s for s in kept if (r, s) in Tlog]
        x = [math.log(max(Tlog[("main", s)][0], 8)) - math.log(max(Tlog[(r, s)][0], 8)) for s in pair]
        m, lo, hi = tint(x)
        s6[r] = {"R": math.exp(m), "lo": math.exp(lo), "hi": math.exp(hi),
                 "n_cens": sum(Tlog[(a, s)][1] for a in ("main", r) for s in pair)}
    out["S6_logit"] = s6
    out["T_table"] = {a: {s: T[(a, s)] for s in seeds if (a, s) in T} for a in arms}
    return out, runs, T


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, runs, T = rebuild(args[0], args[1] if len(args) > 1 else None)
    js = None
    if "--json" in sys.argv:
        js = sys.argv[sys.argv.index("--json") + 1]
        json.dump(out, open(js, "w"), indent=1, default=float)
    print("T table:", json.dumps(out["T_table"]))
    print("mismatch vs stored:", [k for k, v in out["tails"].items() if not v["match"]])
    for r, v in out["ratios"].items():
        print(f"{r:6s} R {v['R']:.3f} [{v['R_lo']:.3f}, {v['R_hi']:.3f}] usable {v['usable']} cens main/ref {v['n_cens_main']}/{v['n_cens_ref']} geomean T {v['geomean_T']}")
    print("port gate", out["port_gate"], "| status (before G1/G2):", out["status_before_G1G2"], out["notes"])
    print("prereg 3of3", out["prereg_3of3"], "5of5", out["prereg_5of5"])
    print("S1", out["S1_law"], "| S2", out["S2"])
    print("S3 D0", {a: [round(x, 4) for x in v] for a, v in out["S3_D0"].items()})
    print("S5 px_off t0", {a: [round(x, 4) for x in v] for a, v in out["S5_pxoff_t0"].items()})
    print("S6 logit", out["S6_logit"])
