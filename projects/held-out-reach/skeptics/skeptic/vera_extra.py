"""Vera: robustness of the NO, run accounting, v1 (scout) prereg scoring, sandbox outcomes."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vera_check import PROJ, MODELS, ROUTE_TOOLS, load_runs, newcombe  # noqa: E402
from vera_consts import consts  # noqa: E402

C = consts()
runs = load_runs()
by = collections.defaultdict(list)
for r in runs:
    by[(r["model"], r["tool"])].append(r)
for k in by:
    by[k].sort(key=lambda r: r["i"])
N = 55


def score(pred_fn, tools=ROUTE_TOOLS, cal_n=None):
    gaps, miss = [], 0
    for t in tools:
        for m in MODELS:
            v = by[(m, t)][:N]
            k = sum(r["log"] for r in v)
            p = pred_fn(m)
            gaps.append(k / N - p)
            if cal_n:
                kc, nc = cal_n(m)
                lo, hi = newcombe(k, N, kc, nc)
                miss += lo > 0.10 or hi < -0.10
    return np.mean(np.abs(gaps)), miss


cal = lambda m: C[m]["cal_rate"]
calkn = lambda m: C[m]["cal"]
print("1) drop-one-tool (frozen calibration):")
for drop in ROUTE_TOOLS:
    keep = [t for t in ROUTE_TOOLS if t != drop]
    mae, miss = score(cal, keep, calkn)
    st = "YES" if (mae <= 0.10 and miss <= 1) else "NO" if (mae > 0.15 or miss >= 2) else "NS"
    print(f"   without {drop:7s}: MAE {mae:.3f}, clear misses {miss} -> {st}")
for t in ROUTE_TOOLS:
    mae, miss = score(cal, [t], calkn)
    print(f"   only {t:7s}: MAE {mae:.3f}, clear misses {miss}")

print("2) calibration from tonight's MCP fetch rerun instead of 1109:")
re = {m: (sum(r["log"] for r in by[(m, "mcpfetch")]), len(by[(m, "mcpfetch")])) for m in MODELS}
print("   rerun", re, "vs 1109", {m: C[m]["cal"] for m in MODELS})
for m in MODELS:
    lo, hi = newcombe(re[m][0], re[m][1], *C[m]["cal"])
    print(f"   {m}: rerun - 1109 = {re[m][0]/re[m][1]-C[m]['cal_rate']:+.3f} newcombe [{lo:+.3f},{hi:+.3f}]")
mae, miss = score(lambda m: re[m][0] / re[m][1], ROUTE_TOOLS, lambda m: re[m])
print(f"   MAE with rerun calibration {mae:.3f}, clear misses {miss}")

print("3) v1 scout prereg (prediction = reach x stuck rate, one-factor):")
hits = tot = 0
for t in ROUTE_TOOLS:
    for m in MODELS:
        v = by[(m, t)][:N]
        g = sum(r["log"] for r in v) / N - C[m]["s_rate"]
        hits += abs(g) <= 0.10
        tot += 1
r0 = sum(r["log"] for m in MODELS for r in by[(m, "wbl")])
print(f"   held-out route combos within 10 points of reach x stuck: {hits}/{tot}; reach-0 follows {r0}/"
      f"{sum(len(by[(m,'wbl')]) for m in MODELS)}")
fr = {m: (sum(r["log"] for r in by[(m, "markitdown")]), len(by[(m, "markitdown")])) for m in MODELS}
print(f"   FREE control markitdown (pred = stuck rate ~0.98-1.00): {fr}")

print("4) run accounting:")
fails = collections.Counter()
fp = os.path.join(PROJ, "raw/failed.jsonl")
if os.path.exists(fp):
    for line in open(fp):
        try:
            f = json.loads(line)
        except Exception:
            continue
        if f.get("phase", "counted") == "counted":
            fails[(f.get("model"), f.get("tool"))] += 1
for t in ROUTE_TOOLS + ["wbl", "mcpfetch", "markitdown"]:
    for m in MODELS:
        idx = [r["i"] for r in by[(m, t)]]
        gaps = sorted(set(range(max(idx) + 1)) - set(idx)) if idx else []
        print(f"   {t:10s} {m:24s} n={len(idx):3d} missing idx {gaps} failed.jsonl {fails[(m, t)]}"
              f" parse_fail runs {sum(1 for r in by[(m,t)] if r['parse_fail'])}")

print("5) reach-shuffle null (12 scored cells: 9 route reach=1, 3 WebBaseLoader reach=0; shuffle reach labels):")
import itertools as _it
cells = [(m, t, 1) for t in ROUTE_TOOLS for m in MODELS] + [(m, "wbl", 0) for m in MODELS]
meas = {(m, t): sum(r["log"] for r in by[(m, t)][:N if t != "wbl" else 30]) / (N if t != "wbl" else 30) for m, t, _ in cells}
def mae_for(labels):
    return np.mean([abs(meas[(m, t)] - (C[m]["cal_rate"] if lab else 0.0)) for (m, t, _), lab in zip(cells, labels)])
true_labels = [c[2] for c in cells]
obs = mae_for(true_labels)
null = [mae_for(list(p)) for p in set(_it.permutations(true_labels))]
print(f"   observed MAE over 12 cells {obs:.3f}; {len(null)} distinct shuffles: min {min(null):.3f} median {np.median(null):.3f};"
      f" share of shuffles <= observed {np.mean(np.array(null) <= obs + 1e-12):.3f}")
