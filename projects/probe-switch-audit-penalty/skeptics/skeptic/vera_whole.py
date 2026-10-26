"""Vera: clause-1/clause-4 cells from build's saved whole-graph rows (own code).
Per run: argmax switch (probed top A AND unprobed top B), strict switch (lo_p > 0 AND lo_u < 0), ties, unprobed-B,
mean push, mean dP(A) over the A/B pair, Wilson 95% interval for the strict rate, and the margin histogram.
Usage: python vera_whole.py WHOLE_DIR RUN [RUN ...]
"""
import sys, json, math, collections

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

sig = lambda x: 1 / (1 + math.exp(-x))
wd = sys.argv[1]
for run in sys.argv[2:]:
    d = json.load(open(f"{wd}/{run}__emu0.json"))
    rows = [r for r in d["rows"] if r["split"] == "test"]
    P = {r["prompt"]: r for r in rows if r["probe"] == 1}; U = {r["prompt"]: r for r in rows if r["probe"] == 0}
    ks = sorted(P); assert ks == sorted(U) and len(ks) == 128, len(ks)
    A, B = rows[0]["a_id"], rows[0]["b_id"]
    # lo recomputed from logp so a stored 'lo' field is not trusted blindly
    lo = {c: {k: (X[k]["logp_a"] - X[k]["logp_b"]) for k in ks} for c, X in (("p", P), ("u", U))}
    assert all(abs(lo["p"][k] - P[k]["lo"]) < 1e-6 for k in ks)
    n = len(ks)
    am = sum(P[k]["top"] == A and U[k]["top"] == B for k in ks)
    st = sum(lo["p"][k] > 0 and lo["u"][k] < 0 for k in ks)
    tp = sum(lo["p"][k] == 0 for k in ks); tu = sum(lo["u"][k] == 0 for k in ks)
    ub = sum(U[k]["top"] == B for k in ks); ub_strict = sum(lo["u"][k] < 0 for k in ks)
    push = sum(lo["p"][k] - lo["u"][k] for k in ks) / n
    dpa = sum(sig(lo["p"][k]) - sig(lo["u"][k]) for k in ks) / n
    other = sum(P[k]["top"] not in (A, B) for k in ks) + sum(U[k]["top"] not in (A, B) for k in ks)
    lo_w, hi_w = wilson(st, n)
    print(f"{run}: argmax switch {am}/{n}={am/n:.3f}  STRICT {st}/{n}={st/n:.3f} (Wilson95 {lo_w:.3f}-{hi_w:.3f})  "
          f"ties probed {tp} unprobed {tu}  unprobed-B argmax {ub/n:.3f} strict {ub_strict/n:.3f}  push {push:.4f}  "
          f"mean dP(A) {dpa:.4f}  top outside A/B {other}  repeat-changed {d['info'].get('cond0_repeat_changed_prompts')},"
          f"{d['info'].get('cond1_repeat_changed_prompts')}")
    print("   lo_p hist", sorted(collections.Counter(round(v, 4) for v in lo["p"].values()).items()))
    print("   lo_u hist", sorted(collections.Counter(round(v, 4) for v in lo["u"].values()).items()))
