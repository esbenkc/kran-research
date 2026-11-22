#!/usr/bin/env python3
"""Vera (skeptic), continuation room oracle-1122-frozen-heads-8b.

Checks on the landed rows that vera_check.py does not do:
 1. copy detection: per-prompt logits_sha256 overlap between seeds/arms (a copied or
    re-scored row would share hashes) and the per-prompt push correlation;
 2. permutation null: flip the probe label of each pair at random (10,000 draws) and
    recompute the strict switch; the observed value must sit far outside;
 3. head-audit consistency from data/audit/<run>.json: R_blind3 = 1 - push_after/push at
    k=3, null p95 from the 50 stored random 3-sets, and the frozen read.
Usage: python3 vera_cont.py DATA_DIR
"""
import json, random, sys, statistics as st
from pathlib import Path

D = Path(sys.argv[1])


def rows(run):
    d = json.loads((D / "whole" / f"{run}.json").read_text())
    by = {}
    for r in d["rows"]:
        by.setdefault(r["prompt"], {})[r["probe"]] = r
    return by


def strict_of(on, off):
    # strict: probed top-1 is A strictly above all others; unprobed top-1 is B strictly
    a_ok = on["top"] == on["a_id"] and on["top_logit"] > on["top2_logit"]
    b_ok = off["top"] == off["b_id"] and off["top_logit"] > off["top2_logit"]
    return a_ok and b_ok


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else float("nan")


def push(r):
    return r["logp_a"] - r["logp_b"]


runs = sorted(p.stem for p in (D / "whole").glob("*.json"))
R = {r: rows(r) for r in runs}

print("== 1. copy detection (shared per-prompt logits sha, probed+unprobed)")
for i, a in enumerate(runs):
    for b in runs[i + 1:]:
        sa = {(p, k): v["logits_sha256"] for p, d in R[a].items() for k, v in d.items()}
        sb = {(p, k): v["logits_sha256"] for p, d in R[b].items() for k, v in d.items()}
        common = [key for key in sa if key in sb]
        same = sum(sa[k] == sb[k] for k in common)
        pa = [push(R[a][p][1]) - push(R[a][p][0]) for p in sorted(R[a])]
        pb = [push(R[b][p][1]) - push(R[b][p][0]) for p in sorted(R[b])]
        c = pearson(pa, pb)
        print(f"{a:18s} {b:18s} same-sha {same:3d}/{len(common)}  push corr {c:+.3f}")

print("== 2. probe-label permutation null (10,000 draws, rng 0)")
rng = random.Random(0)
for run in runs:
    prompts = sorted(R[run])
    obs = sum(strict_of(R[run][p][1], R[run][p][0]) for p in prompts) / len(prompts)
    ge = 0
    draws = []
    for _ in range(10000):
        s = 0
        for p in prompts:
            on, off = (R[run][p][1], R[run][p][0]) if rng.random() < 0.5 else (R[run][p][0], R[run][p][1])
            s += strict_of(on, off)
        v = s / len(prompts)
        draws.append(v)
        ge += v >= obs
    draws.sort()
    print(f"{run:18s} obs strict {obs:.4f}  null mean {st.mean(draws):.4f} p99.9 {draws[9989]:.4f} max {draws[-1]:.4f}  P(null>=obs) {(ge+1)/10001:.2e}")

print("== 3. head audit (data/audit)")
for f in sorted((D / "audit").glob("*.json")):
    a = json.loads(f.read_text())
    k3 = [x for x in a["blind_greedy_test"] if x["k"] == 3][0]
    r3 = 1 - k3["push_after_mean"] / k3["push_mean"]
    nl = sorted(a["null_removed"])
    # p95 over 50 values: report both nearest-rank and linear interpolation
    nr = nl[min(len(nl) - 1, int(0.95 * len(nl) + 0.999999) - 1)]
    pos = 0.95 * (len(nl) - 1)
    lin = nl[int(pos)] + (nl[min(int(pos) + 1, len(nl) - 1)] - nl[int(pos)]) * (pos - int(pos))
    read = ("audit finds the frozen carrier (>=0.70)" if r3 >= 0.70 else
            "audit fails (<0.30)" if r3 < 0.30 else "between")
    heads = k3["heads"]
    stored = a["R_blind3"]
    print(f"{a['run']:18s} heads {heads} R_blind3 {r3:.4f} (stored {stored:.4f}, diff {r3-stored:+.1e}) "
          f"null p95 nearest {nr:.4f} linear {lin:.4f} (stored {a['null_p95']:.4f}) max {nl[-1]:.4f} -> {read}")
    print("   greedy test k=1..3 removed:", [round(x["removed"], 4) for x in a["blind_greedy_test"]],
          " clamp-all removed", round(a["clamp_all_test"]["removed"], 4))
