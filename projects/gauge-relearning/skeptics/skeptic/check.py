"""skeptic: independent recompute of steps-to-recover from raw runs.jsonl.

Usage: python check.py <runs.jsonl> <meta.json> [threshold]
Own code; does not import anything from the build project.
"""
import json, math, sys, random, itertools
from collections import defaultdict
from statistics import median

import glob
runs_path, meta_path = sys.argv[1], sys.argv[2]
meta = json.load(open(meta_path))
allrows = [json.loads(l) for f in sorted(glob.glob(runs_path, recursive=True)) for l in open(f) if l.strip()]
# dedupe (raw files + collected file could overlap)
seen, rows0 = set(), []
for r in allrows:
    k = (r.get("model"), r.get("order", 0), r["attacker"], r["log2c"], r["seed"], r["lr"], r["step"])
    if k not in seen:
        seen.add(k); rows0.append(r)
MODEL = "unlearned"
rows = [r for r in rows0 if r.get("model", MODEL) == MODEL and r.get("order", 0) == 0]
order_rows = [r for r in rows0 if r.get("model", MODEL) == MODEL and r.get("order", 0) != 0]
floor_rows = [r for r in rows0 if r.get("model") == "floor"]
print(f"rows={len(rows)} floor_rows={len(floor_rows)} keys={sorted(rows0[0].keys())}")


def thr_from_meta(m):
    for k in ("threshold_acc_A", "threshold", "recover_threshold", "recovery_threshold"):
        if k in m:
            v = m[k]
            return v if isinstance(v, (int, float)) else v.get("acc_B", v)
    raise SystemExit(f"no threshold in meta: {list(m)}")


ACC = sys.argv[4] if len(sys.argv) > 4 else "acc_A"
THR = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else thr_from_meta(meta)
CAP = meta.get("max_steps", meta.get("cap", meta.get("step_cap", 500)))
print(f"threshold {ACC}>={THR}  cap={CAP}")

runs = defaultdict(list)
for r in rows:
    key = (r["attacker"], round(r["log2c"]), r["seed"], r["lr"])
    runs[key].append(r)
fl = defaultdict(list)
for r in floor_rows:
    fl[(r["attacker"], round(r["log2c"]), r["seed"], r["lr"])].append(r)


def outcome(trace):
    trace = sorted(trace, key=lambda r: r["step"])
    for r in trace:
        loss = r.get("loss")
        if loss is not None and not math.isfinite(loss):
            return ("diverged", r["step"])
        if r[ACC] >= THR:
            return ("recovered", r["step"])
    return ("censored", max(r["step"] for r in trace))


res = {k: outcome(v) for k, v in runs.items()}
# step-0 identity across c: the gauge must not change the starting model's outputs
s0 = defaultdict(set)
for k, v in runs.items():
    r0 = [r for r in v if r["step"] == 0]
    if r0:
        s0[(k[0], k[2])].add((k[1], r0[0]["acc_A"], round(r0[0]["loss"], 4)))
print("STEP-0 identity across c (attacker,seed): distinct (acc_A, loss) values")
for k, vals in sorted(s0.items()):
    d = {(a, l) for _, a, l in vals}
    print(f"  {k}: {len(d)} distinct {sorted(d)[:3]}")
fres = {k: outcome(v) for k, v in fl.items()}
attackers = sorted({k[0] for k in res})
cs = sorted({k[1] for k in res})
seeds = sorted({k[2] for k in res})
print("attackers", attackers, "log2c", cs, "seeds", seeds)


def score(o):  # for best-of-grid: recovered < censored < diverged
    st, s = o
    return (0, s) if st == "recovered" else (1, 0) if st == "censored" else (2, 0)


best = {}
for a in attackers:
    for c in cs:
        cand = [(k[3], res[k]) for k in res if k[0] == a and k[1] == c and k[2] == seeds[0]]
        if not cand:
            continue
        lr, o = min(cand, key=lambda t: (score(t[1]), t[0]))
        best[(a, c)] = lr

table = {}
print("\nattacker  log2c  best_lr   per-seed outcomes (seed:status@step)")
for a in attackers:
    for c in cs:
        if (a, c) not in best:
            continue
        lr = best[(a, c)]
        grid = sorted({k[3] for k in res if k[0] == a and k[1] == c and k[2] == seeds[0]})
        edge = " GRID-EDGE" if lr in (grid[0], grid[-1]) else ""
        outs = {s: res.get((a, c, s, lr)) for s in seeds}
        table[(a, c)] = outs
        txt = " ".join(f"{s}:{o[0][0]}@{o[1]}" if o else f"{s}:MISSING" for s, o in outs.items())
        print(f"{a:10s} {c:+3d}  {lr:<9g} {txt}{edge}  grid={grid}")


def steps(o):  # censored counted at cap (lower bound), diverged -> None
    if o is None or o[0] == "diverged":
        return None
    return o[1] if o[0] == "recovered" else CAP


print("\nmedian steps (censored=cap) per attacker x log2c")
summ = {}
for a in attackers:
    line = []
    for c in cs:
        v = [steps(o) for o in table.get((a, c), {}).values()]
        v = [x for x in v if x is not None]
        summ[(a, c)] = v
        line.append(f"{c:+d}:{median(v) if v else 'NA'}[{min(v) if v else ''}-{max(v) if v else ''}]")
    print(f"{a:10s}", "  ".join(line))


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


print("\nNULL: label-shuffle permutation test, statistic = Spearman(log2c, log steps) per side")
rng = random.Random(12345)
for a in attackers:
    for side, sel in (("c>=1", lambda c: c >= 0), ("c<=1", lambda c: c <= 0)):
        xs, ys = [], []
        for c in cs:
            if not sel(c):
                continue
            for v in summ.get((a, c), []):
                xs.append(abs(c)); ys.append(math.log(max(v, 1)))
        if len(set(xs)) < 2:
            continue
        obs = spearman(xs, ys)
        n_ge = 0
        N = 20000
        for _ in range(N):
            p = ys[:]; rng.shuffle(p)
            if abs(spearman(xs, p)) >= abs(obs) - 1e-12:
                n_ge += 1
        print(f"{a:10s} {side}: rho={obs:+.3f}  perm p={(n_ge+1)/(N+1):.4f}  n={len(xs)}")

print("\nNULL: c=1 seed spread (noise floor) and fold change of extremes vs c=1")
for a in attackers:
    base = summ.get((a, 0), [])
    if not base:
        continue
    lo, hi = min(base), max(base)
    for c in cs:
        v = summ.get((a, c), [])
        if not v or c == 0:
            continue
        sep = "ABOVE" if min(v) > hi else "BELOW" if max(v) < lo else "overlap"
        print(f"{a:10s} log2c={c:+d} steps {sorted(v)} vs c=1 {sorted(base)} -> {sep} band, fold(median)={median(v)/median(base):.2f}")

print("\nSELECTION BIAS: seed0 (lr picked on it) vs other seeds at chosen lr")
d = []
for (a, c), outs in table.items():
    s0 = steps(outs.get(seeds[0]))
    others = [steps(outs.get(s)) for s in seeds[1:]]
    others = [x for x in others if x is not None]
    if s0 is not None and others:
        d.append(math.log(s0) - math.log(median(others)))
if d:
    print(f"mean log(seed0/others)={sum(d)/len(d):+.3f} over {len(d)} cells (negative = seed0 optimistic)")

print("\nFLOOR control (never-knew model): best lr outcome per attacker/c/seed")
for a in sorted({k[0] for k in fres}):
    for c in sorted({k[1] for k in fres if k[0] == a}):
        for sd in sorted({k[2] for k in fres if k[0] == a and k[1] == c}):
            cand = [(k[3], o) for k, o in fres.items() if k[:3] == (a, c, sd)]
            lr, o = min(cand, key=lambda t: (score(t[1]), t[0]))
            print(f"floor {a:10s} log2c={c:+d} seed={sd} best lr={lr:g} -> {o}  (all: {sorted(cand)})")

print("\nEXTRA ORDERS (paired ratio c/c1 within seed+order, incl. main order 0)")
oruns = defaultdict(list)
for r in order_rows + rows:
    oruns[(r["attacker"], round(r["log2c"]), r["seed"], r.get("order", 0), r["lr"])].append(r)
ores = {k: outcome(v) for k, v in oruns.items()}
for a in attackers:
    for c in cs:
        if c == 0 or (a, c) not in best:
            continue
        ratios = []
        for (a2, c2, sd, od, lr), o in ores.items():
            if a2 != a or c2 != c or lr != best[(a, c)]:
                continue
            base = ores.get((a, 0, sd, od, best.get((a, 0))))
            if base and steps(o) and steps(base):
                ratios.append((sd, od, round(steps(o) / steps(base), 2)))
        if len({x[1] for x in ratios}) > 1:
            rr = [x[2] for x in ratios]
            print(f"{a:7s} log2c={c:+d} n={len(rr)} median={median(rr):.2f} frac>1={sum(x > 1 for x in rr)/len(rr):.2f} {sorted(ratios)}")

json.dump({f"{a}|{c}": v for (a, c), v in summ.items()}, open("skeptic_summary.json", "w"), indent=1)
