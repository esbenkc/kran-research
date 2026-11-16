"""Vera T3 check: the ported rl.py (old recipe, Qwen3-0.6B) vs the parent's rd_p0 runs, curve by curve."""
import json, subprocess, sys
R = "/Users/esben/work/theory-1116-cue-size-law-8b/kran-research"
P = "/Users/esben/work/theory-1116-cue-size-law-8b/swarm-theory-1116-cue-size-law-8b/skeptic/parent_runs"


def curve(d):
    c = {e["step"]: e for e in d["dev_curve"]}
    for e in d.get("dense_curve", []) or []:
        c.setdefault(e["step"], e)
    return c


def T2(c, thr=0.10):
    s = sorted(c)
    g0 = c[0]["gap"]
    for i in range(len(s) - 1):
        if c[s[i]]["gap"] - g0 >= thr and c[s[i + 1]]["gap"] - g0 >= thr:
            return s[i]


for new in ["smoke/port_06b/p0_a0.03125_s0.json", "smoke/port_06b/p0_a0.03125_s1.json", "smoke/runs/p0_a0.03125_s0_v2.json"]:
    n = json.load(open(f"{R}/projects/cue-size-law-8b/{new}"))
    seed = n["args"]["seed"]
    p = json.load(open(f"{P}/p0_a0.03125_s{seed}.json"))
    cn, cp = curve(n), curve(p)
    common = sorted(set(cn) & set(cp))
    maxdiff = max(max(abs(cn[s]["px_off"] - cp[s]["px_off"]), abs(cn[s]["px_on"] - cp[s]["px_on"])) for s in common)
    tg = max(abs(a - b) for a, b in zip(n["test"]["px_off"], p["test"]["px_off"])) if "test" in n and "test" in p else None
    print(new, "seed", seed, "evals new/parent/common", len(cn), len(cp), len(common),
          "max |dP| over common evals", maxdiff, "T new/parent (my rule, all evals)", T2(cn), T2(cp),
          "saved T", n.get("T"), p.get("T"), "max|d test px_off|", tg)
