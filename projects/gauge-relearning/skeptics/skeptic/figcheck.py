"""skeptic: recompute the figure's plotted ratios from raw files, compare to data.json.
Ratio = steps(attacker, c, seed, order) / min over attackers of steps(c=1, same seed, order).
lr per (attacker, c) = fastest on seed 0 order 0 (ties -> lower lr), reused elsewhere.
python figcheck.py <raw_glob> <data.json> [threshold]"""
import glob, json, math, sys
from collections import defaultdict
from statistics import median

THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
runs = defaultdict(list)
for f in [g for g in glob.glob(sys.argv[1]) if "devcpu_" not in g]:
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            if r.get("model", "unlearned") != "unlearned":
                continue
            runs[(r["attacker"], r["log2c"], r["seed"], r.get("order", 0), r["lr"])].append(r)


def hit(rows):
    for r in sorted(rows, key=lambda r: r["step"]):
        if r["loss"] is not None and not math.isfinite(r["loss"]):
            return None  # diverged
        if r["acc_A"] >= THR:
            return r["step"]
    return math.inf  # censored


H = {k: hit(v) for k, v in runs.items()}
best = {}
for (a, c, s, o, lr), h in H.items():
    if s == 0 and o == 0 and h is not None:
        cur = best.get((a, c))
        if cur is None or (h, lr) < (H[(a, c, 0, 0, cur)], cur):
            best[(a, c)] = lr

atk = sorted({k[0] for k in best} - {"freezeqk"})
pairs = sorted({(k[2], k[3]) for k in H})
base = {}
for s, o in pairs:
    v = [H.get((a, 0, s, o, best.get((a, 0)))) for a in atk]
    v = [x for x in v if x is not None]
    if v:
        base[(s, o)] = min(v)
data = json.load(open(sys.argv[2]))
print("base mine vs fig:", {f"seed {s}, order {o}": b for (s, o), b in base.items()} == {k: v for k, v in data["base_steps_by_run"].items()}, base)
fig = {(r["attacker"], r["log2c"]): r for r in data["rows"] if r["kind"] == "tuned"}
for a in atk:
    for c in sorted({k[1] for k in best if k[0] == a}):
        rs = [H[(a, c, s, o, best[(a, c)])] / base[(s, o)] for s, o in pairs
              if (a, c, s, o, best[(a, c)]) in H and H[(a, c, s, o, best[(a, c)])] is not None and (s, o) in base]
        if not rs:
            continue
        f = fig.get((a, c), {})
        mine = (round(median(rs), 3), round(min(rs), 3), round(max(rs), 3), len(rs))
        theirs = (f.get("ratio_median"), f.get("ratio_min"), f.get("ratio_max"), f.get("n_runs"))
        ok = all(abs(x - y) < 0.01 for x, y in zip(mine[:3], theirs[:3]) if y is not None) and mine[3] == theirs[3]
        print(f"{'OK ' if ok else 'DIFF'} {a:7s} log2c={c:+d} mine(med,min,max,n)={mine} fig={theirs}")
