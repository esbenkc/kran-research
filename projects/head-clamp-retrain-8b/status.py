"""Apply the PREREG FROZEN RULE mechanically to whatever cells exist -> data/status.json, proof-matrix.md.

Per retrain cell: back = RETURNED within the budget, or test strict >= 0.25 at the budget; held = test strict < 0.25 at
the budget; CONVERGED (parent rule, verbatim): |mean loss over the last 20% of updates - mean over the 20% before|
<= 0.02 nats AND dev strict over the last 3 evals within a range of 0.10. Missing cells -> 'missing'.
Title status, prereg outcome, return ratio read by seeds vs data/band_17b.json, seed-level 95% t-intervals.

    $PY status.py        (stdlib + scipy)
"""
import json
import math
import os
from pathlib import Path

from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / os.environ.get("HC_OUT", "data")


def load(p):
    return json.load(open(p)) if p.exists() else None


def converged(r) -> bool:
    L = r["loss"]
    n = len(L)
    k = max(1, n // 5)
    if n < 2 * k:
        return False
    a, b = sum(L[n - k:]) / k, sum(L[n - 2 * k:n - k]) / k
    dev = [e["switch_correct"] for e in r["dev_curve"][-3:]]
    return abs(a - b) <= 0.02 and max(dev) - min(dev) <= 0.10


def tint(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return {"mean": m, "low": m - h, "high": m + h, "n": n, "method": f"95% t-interval over {n} seeds (df {n - 1})"}


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def main() -> None:
    sw = {}
    for s in range(5):
        r = load(OUT / "runs" / f"switch_s{s}.json")
        if r:
            sw[s] = {"t_f": r["t_f"], "formed": r["formed"], "steps_run": r["steps_run"],
                     "final_test_strict": r["final_test_strict"]}
    formed = [s for s in sorted(sw) if sw[s]["formed"]][:3]
    cells = {}
    for f in sorted((OUT / "runs").glob("*.json")) if (OUT / "runs").exists() else []:
        r = json.load(open(f))
        if r["mode"] != "clamp":
            continue
        s = r["args"]["seed"]
        t = r["test_merged_clamp"]
        n = t["n_prompts"]
        k = round(t["switch_correct"] * n)
        back = r["stop_reason"] == "returned" or t["switch_correct"] >= 0.25
        cells[r["name"]] = {"arm": r["args"]["arm"], "seed": s, "n_heads": len(r["heads"]),
                            "t0_test_strict": r["t0"]["test_clamp"]["switch_correct"],
                            "stop_reason": r["stop_reason"], "steps_run": r["steps_run"], "budget": r["args"]["steps"],
                            "return_step": r["return_step"], "final_test_strict": t["switch_correct"],
                            "final_test_wilson95": wilson(k, n), "back": back, "converged": converged(r),
                            "max_loss_last10pct": max(r["loss"][-max(1, len(r["loss"]) // 10):]) if r["loss"] else None,
                            "dev_strict_last3": [e["switch_correct"] for e in r["dev_curve"][-3:]],
                            "t_f": sw.get(s, {}).get("t_f"),
                            "ratio": (r["return_step"] / sw[s]["t_f"]) if (r["return_step"] and sw.get(s, {}).get("t_f")) else None}

    def arm(a):
        return [cells.get(f"{a}_s{s}") for s in formed]
    allname = "all64" if any(c["arm"] == "all64" for c in cells.values()) else "all32"
    st = arm("set")
    valid = [c is not None and c["t0_test_strict"] < 0.25 for c in st]
    n_back = sum(1 for c in st if c and c["back"])
    n_held_conv = sum(1 for c in st if c and not c["back"] and c["converged"])
    if len(formed) < 3 or not all(valid):
        status = "UNCONFIRMED"
        why = f"{len(formed)} formed seeds" if len(formed) < 3 else "SET not removed at t=0 or cell missing"
    elif n_back >= 2:
        status, why = "REJECTED", f"SET back in {n_back} of 3 seeds"
    elif n_held_conv >= 2:
        status, why = "CONFIRMED", f"SET held (converged) in {n_held_conv} of 3 seeds"
    else:
        status, why = "UNCONFIRMED", "neither rule met"

    def cnt(a, pred):
        return sum(1 for c in arm(a) if c and pred(c))
    set_by_tf = cnt("set", lambda c: c["stop_reason"] == "returned" and c["return_step"] <= c["t_f"])
    held = (set_by_tf >= 2 and cnt("top1", lambda c: c["stop_reason"] == "returned") >= 2
            and cnt("leave1", lambda c: c["stop_reason"] == "returned") >= 2
            and cnt(allname, lambda c: not c["back"]) >= 2)
    failed = (cnt("set", lambda c: not (c["stop_reason"] == "returned" and c["return_step"] <= c["t_f"])) >= 2
              or cnt("top1", lambda c: c["stop_reason"] != "returned") >= 2
              or cnt("leave1", lambda c: c["stop_reason"] != "returned") >= 2
              or cnt(allname, lambda c: c["back"]) >= 2)
    outcome = "held" if held else ("failed" if failed else "unsettled")
    band = load(OUT / "band_17b.json")
    ratio_read = None
    if band and all(c is not None for c in st) and len(st) == 3:
        lo, hi = band["band"]
        rs = [c["ratio"] if c["stop_reason"] == "returned" else math.inf for c in st]
        below, inside, above = (sum(r < lo for r in rs), sum(lo <= r <= hi for r in rs), sum(r > hi for r in rs))
        ratio_read = {"band": band["band"], "ratios": [None if r == math.inf else r for r in rs],
                      "read": "FASTER" if below >= 2 else "SLOWER" if above >= 2 else
                      "SIZE-INVARIANT" if inside >= 2 else "UNRESOLVED",
                      "t_interval": tint([r for r in rs if r != math.inf])}
    out = {"rule": "PREREG.md FROZEN RULE", "formed_seeds": formed, "switch": sw, "status": status, "why": why,
           "prereg_outcome": outcome, "set_returned_by_T_f": set_by_tf, "ratio_read": ratio_read,
           "headline_seed_level": tint([c["final_test_strict"] for c in st if c]),
           "cells": cells}
    (OUT / "status.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("formed_seeds", "status", "why", "prereg_outcome", "ratio_read",
                                          "headline_seed_level")}))
    for name, c in cells.items():
        print(f"{name:12s} n={c['n_heads']:2d} t0 {c['t0_test_strict']:.3f} {c['stop_reason']:8s} "
              f"steps {c['steps_run']:5d}/{c['budget']} ret {c['return_step']} test {c['final_test_strict']:.3f} "
              f"back {c['back']} conv {c['converged']}")


if __name__ == "__main__":
    main()
