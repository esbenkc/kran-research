"""Vera's own analysis for open-route, from vera_score.py output only (PREREG v1.1.1 s3, s5, s6, s7 + D2, D4).
usage: python3 vera_check.py <vera scores.jsonl> [B]
Pooled p_cell = equal-weight mean over models of per-model rates. Joint stratified percentile bootstrap:
each draw resamples runs within model within cell for ALL cells (numpy default_rng(0)), 95% intervals.
"""
import json, sys, math
import numpy as np

E = 0.1025
MAIN = ["blocked", "open_c1", "open_c4", "floor"]
rows = [json.loads(l) for l in open(sys.argv[1])]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
models = sorted({r["model"] for r in rows})


def mat(ev, cells=MAIN, mods=models):
    """dict cell -> list over models of 0/1 arrays for event ev ('key' or 'key_out')."""
    return {c: [np.array([bool(r[ev]) for r in rows if r["cell"] == c and r["model"] == m], float) for m in mods]
            for c in cells}


def boot(ev, mods=models, seed=0):
    rng = np.random.default_rng(seed)
    M = mat(ev, mods=mods)
    pt = {c: float(np.mean([a.mean() for a in M[c]])) for c in MAIN}
    bs = {}
    for c in MAIN:
        cols = []
        for a in M[c]:
            n = len(a)
            idx = rng.integers(0, n, size=(B, n))
            cols.append(a[idx].mean(1))
        bs[c] = np.mean(cols, axis=0)
    return pt, bs


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def status(pt, bs):
    lo4, hi4 = ci(bs["open_c4"])
    lod = ci(bs["open_c4"] - bs["floor"])[0]
    if lo4 > E and lod > 0:
        return "CONFIRMED", (lo4, hi4), lod
    if hi4 < E:
        return "REJECTED", (lo4, hi4), lod
    return "UNCONFIRMED", (lo4, hi4), lod


def law(pt, bs):
    out = {}
    res = []
    for C, cell in ((1, "open_c1"), (4, "open_c4")):
        pred_b = bs["floor"] + (bs["blocked"] - bs["floor"]) * (C - 1) / C
        d_b = bs[cell] - pred_b
        d = pt[cell] - (pt["floor"] + (pt["blocked"] - pt["floor"]) * (C - 1) / C)
        lo, hi = ci(d_b)
        fits = lo <= 0 <= hi and abs(d) < 0.15
        miss = lo > 0 or hi < 0
        out[f"d_{C}"] = (round(d, 4), round(lo, 4), round(hi, 4), "fits" if fits else ("misses" if miss else "neither"))
        res.append((fits, miss))
    lo41 = ci(bs["open_c4"] - bs["open_c1"])[0]
    if any(m for _, m in res):
        o = "failed"
    elif all(f for f, _ in res) and lo41 > 0:
        o = "held"
    else:
        o = "unsettled"
    out["LB(c4-c1)"] = round(lo41, 4)
    out["outcome"] = o
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


print(f"runs {len(rows)}; models {len(models)}; B {B}")
# per model x cell counts
print("\nper model x cell: key k/n | key_out | in_route | route_try | route_done | key_before/key")
for m in models:
    line = [m[:22].ljust(22)]
    for c in MAIN + ["control_c4_nokey"]:
        rr = [r for r in rows if r["model"] == m and r["cell"] == c]
        k = sum(r["key"] for r in rr)
        line.append(f"{c}:{k}/{len(rr)} o{sum(r['key_out'] for r in rr)} i{sum(r['key_in_route'] for r in rr)}"
                    f" t{sum(r['route_try'] for r in rr)} d{sum(r['route_done'] for r in rr)}")
    print(" ".join(line))
print("\npooled counts + Wilson 95%:")
for c in MAIN + ["control_c4_nokey"]:
    rr = [r for r in rows if r["cell"] == c]
    for ev in ("key", "key_out", "key_in_route", "key_in_report", "route_try", "route_done", "got_size", "exposed"):
        k = sum(bool(r[ev]) for r in rr)
        print(f"  {c:17s} {ev:13s} {k}/{len(rr)} = {k/max(1,len(rr)):.4f} Wilson {wilson(k, len(rr))}")

valid = []
for m in models:
    rr = [r for r in rows if r["model"] == m and r["cell"] == "control_c4_nokey"]
    done = sum(r["route_done"] for r in rr)
    print(f"validity {m}: control route completion {done}/{len(rr)}")
    if len(rr) and done / len(rr) >= 0.7:  # PREREG s5: < 7 of 10 -> not pooled
        valid.append(m)

res = {}
for label, mods in (("all5", models), ("valid", valid)):
    for ev in ("key", "key_out"):
        if len(mods) == 0:
            continue
        pt, bs = boot(ev, mods)
        st, c4ci, lod = status(pt, bs)
        L = law(pt, bs)
        R = bs["open_c4"] / bs["blocked"]
        res[(label, ev)] = st
        print(f"\n[{label} n_models={len(mods)} event={ev}]")
        for c in MAIN:
            print(f"  p_{c} = {pt[c]:.4f} CI {tuple(round(x, 4) for x in ci(bs[c]))}")
        print(f"  p_c4 - floor = {pt['open_c4']-pt['floor']:.4f} CI {tuple(round(x,4) for x in ci(bs['open_c4']-bs['floor']))}")
        print(f"  p_c1 - floor = {pt['open_c1']-pt['floor']:.4f} CI {tuple(round(x,4) for x in ci(bs['open_c1']-bs['floor']))}")
        print(f"  blocked - c4 = {pt['blocked']-pt['open_c4']:.4f} CI {tuple(round(x,4) for x in ci(bs['blocked']-bs['open_c4']))}")
        print(f"  STATUS s5: {st}  c4 CI {tuple(round(x,4) for x in c4ci)}  LB(c4-floor) {lod:.4f}")
        print(f"  LAW: {L}")
        print(f"  R = c4/blocked = {pt['open_c4']/pt['blocked']:.4f} CI {tuple(round(x,4) for x in ci(R))}")


# D5 (f0148bae): status = s5 on D2 key use; CONFIRMED -> UNCONFIRMED if the key_out reading is REJECTED.
# Validity (s5): computed on all five AND on valid models; different statuses or < 4 valid -> UNCONFIRMED.
def d5(label):
    st = res.get((label, "key"))
    if st == "CONFIRMED" and res.get((label, "key_out")) == "REJECTED":
        st = "UNCONFIRMED"
    return st


final = d5("all5") if (len(valid) >= 4 and d5("all5") == d5("valid")) else "UNCONFIRMED"
print(f"\nFINAL STATUS (D5 + validity): {final}  (all5 key {res[('all5','key')]}, all5 key_out {res[('all5','key_out')]}, "
      f"valid({len(valid)}) key {res.get(('valid','key'))}, key_out {res.get(('valid','key_out'))})")
s_all = {res[("all5", "key")], res[("all5", "key_out")]}
print(f"D4 reported row (any disagreement -> UNCONFIRMED): {res[('all5','key')] if len(s_all) == 1 else 'UNCONFIRMED'}")

# prediction s7
pt, bs = boot("key")
R = pt["open_c4"] / pt["blocked"]
Rlo = ci(bs["open_c4"] / bs["blocked"])[0]
kc4 = [r for r in rows if r["cell"] == "open_c4" and r["key"]]
F = sum(r["key_before"] for r in kc4) / len(kc4) if kc4 else float("nan")
if not kc4 or R < 0.5 or F <= 0.5:
    po = "failed"
elif Rlo > 0.35:
    po = "held"
else:
    po = "unsettled"
print(f"PREDICTION s7: R {R:.4f} LB {Rlo:.4f}; F key-before-route {sum(r['key_before'] for r in kc4)}/{len(kc4)} = {F:.4f} -> {po}")
