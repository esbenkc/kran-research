"""Vera: status rebuild. (A) recompute build's check-1 scores from the raw per-start MPS reads in
data/heldout_nudge/<m>.json (score = max_u [max_start lift_model(u) - max_start lift_untrained(u)], D4/D12),
(B) my own check-1 scores from my own search (vera_nudge_status*.json, bf16 reads, untrained-subtracted per unit),
(C) check 2 from raw per-prompt P (vera_check.check2), dGap from raw (vera_check.gaps); then my own status rule.
"""
import glob, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_check as vc
import vera_rule as vr
HERE = os.path.dirname(os.path.abspath(__file__))
CLEANS = ["cln_s0", "cln_s1", "cln_s2"]; TWINS = ["stack_s0", "stack_s1"]; SW = ["adv_mid_s0", "adv_mid_s1"]


def build_check1():
    out = {}
    raw = {}
    for f in glob.glob(f"{vc.P}/data/heldout_nudge/*.json"):
        d = json.load(open(f)); m = d["model"]
        per = {}
        for r in d["mps"]["reads"]:
            if str(r["start"]).startswith("r"):  # status uses random starts only
                per[r["ui"]] = max(per.get(r["ui"], -1e9), r["lift"])
        raw[m] = (per, d["score"])
    if "base" not in raw:
        return {}
    b = raw["base"][0]
    for m, (per, sc) in raw.items():
        out[m] = {"score": max(per[u] - b[u] for u in per), "build_score": sc, "n_units": len(per)}
    return out


def my_check1():
    out = {}
    for f, bkey in [("vera_nudge_status.json", "base"), ("vera_nudge_status16.json", "base16")]:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if bkey not in d:
            continue
        b = d[bkey]["units"]
        for m, r in d.items():
            if m.startswith("base"):
                continue
            sc = {u: r["units"][u]["lift_random_bf16"] - b[u]["lift_random_bf16"] for u in r["units"]}
            pl = {u: r["units"][u]["lift_planted_opt_bf16"] - b[u]["lift_planted_opt_bf16"] for u in r["units"]}
            out[m] = {"score": max(sc.values()), "unit": max(sc, key=sc.get), "planted": max(pl.values()), "file": f}
    return out


if __name__ == "__main__":
    g = vc.gaps(); c2 = vc.check2(); b1 = build_check1(); m1 = my_check1()
    gaps = {m: g[m]["dgap"] for m in g}
    print("dGap:", {m: round(v, 4) for m, v in gaps.items()})
    print("check1 build (recomputed from raw reads):", {m: (round(v["score"], 3), round(v["build_score"], 3)) for m, v in b1.items()})
    print("check1 mine:", {m: (round(v["score"], 3), v["unit"], round(v["planted"], 3)) for m, v in m1.items()})
    print("check2:", {m: round(v["score"], 3) for m, v in c2.items()})
    res = {"gaps": g, "check1_build": b1, "check1_mine": m1, "check2": c2}
    for name, s1 in [("build_check1", b1), ("my_check1", m1)]:
        scores = {"nudge": {m: v["score"] for m, v in s1.items()}, "sys": {m: v["score"] for m, v in c2.items()}}
        ev = vr.evaluate(scores, gaps, CLEANS, TWINS, SW, checks=("nudge", "sys"))
        lit = None
        if all(m in scores["nudge"] for m in CLEANS) and all(m in scores["sys"] for m in CLEANS):
            lit = vr.literal_prereg(scores, gaps, ["cln_s0", "cln_s1"], "cln_s2", SW, checks=("nudge", "sys"))
        res[name] = {"eval": ev, "literal": lit}
        print("\n==", name, "status:", ev["status"])
        for c, L in ev["lines"].items():
            print(f"  {c}: line95 {L['line']:.4f} (mean {L['mean']:.4f} sd {L['sd']:.4f}) line99 {L['line99']:.4f} clean max {L['clean_max']:.4f}"
                  f" LOO false alarms {ev['loo'][c]}/3 twins flagged {ev['twin_flag'][c]}")
        print("  informative:", ev["informative"], "| switches:", {s: (round(t['gap'], 3), t['kept_gap'], t['tell']) for s, t in ev["tells"].items()})
        print("  literal prereg:", lit)
    json.dump(res, open(os.path.join(HERE, "vera_status.json"), "w"), indent=1, default=str)
