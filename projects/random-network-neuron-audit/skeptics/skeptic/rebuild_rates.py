"""skeptic: rebuild one-job rates from RAW judge text (label_raw, det_raw) with my own parser.

Verdict per (unit, judge), per frozen prereg:
  one_job  := main_type != NONE and main_count >= 15 and detection hits >= 4
  nonlex   := one_job and main_type != LEXICAL
Unit score = mean of the two judges. Also AND of both judges, each judge alone, Cohen's kappa.
CIs: 95% percentile bootstrap over units (10k). Gap tests: unit-level permutation of arm labels (10k).
Parser sensitivity: FIRST vs LAST complete JSON object in label_raw (haiku sometimes 'reconsiders').
"""
import json
import random
import re
import sys
from collections import defaultdict

PATH = sys.argv[1]
WHICH = sys.argv[2] if len(sys.argv) > 2 else "last"
rows = [json.loads(l) for l in open(PATH)]


def json_objects(text):
    """All top-level {...} objects that json-parse, in order (brace matching, string-aware)."""
    out, depth, start, ins, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if ins:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                ins = False
            continue
        if ch == '"':
            ins = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def parse_label(raw):
    objs = [o for o in json_objects(raw) if "main_count" in o]
    if not objs:
        return None
    o = objs[-1] if WHICH == "last" else objs[0]
    t = str(o.get("main_type", "")).upper().strip()
    try:
        c = int(o.get("main_count"))
    except (TypeError, ValueError):
        return None
    return t, c, o.get("main_label", "")


def parse_hits(raw, truth):
    objs = [o for o in json_objects(raw) if "picks" in o]
    if not objs:
        return None
    picks = []
    for p in objs[-1]["picks"]:
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if 1 <= p <= 10 and p not in picks:
            picks.append(p)
    return len(set(picks[:5]) & set(truth))


v = {}  # (uid, judge) -> (one_job, nonlex, type)
meta = {}
fail = defaultdict(int)
for r in rows:
    meta[r["uid"]] = (r["arm"], r["layer"])
    pl = parse_label(r.get("label_raw") or "")
    if pl is None:
        fail["label"] += 1
        v[(r["uid"], r["judge"])] = (0, 0, "UNPARSED")
        continue
    t, c, _ = pl
    h = parse_hits(r.get("det_raw") or "", r["truth"]) if r.get("det_raw") else None
    if h is None:
        fail["det"] += 1
        h = 0
    one = int(t != "NONE" and c >= 15 and h >= 4)
    v[(r["uid"], r["judge"])] = (one, int(one and t != "LEXICAL"), t)

judges = sorted({j for _, j in v})
units = defaultdict(list)
for uid, (arm, layer) in meta.items():
    units[arm].append(uid)


def score(uid, idx, mode):
    vals = [v[(uid, j)][idx] for j in judges if (uid, j) in v]
    if mode == "mean":
        return sum(vals) / len(vals)
    if mode == "and":
        return float(all(vals))
    return float(v[(uid, mode)][idx]) if (uid, mode) in v else None


def boot(xs, n=10000, seed=0):
    rng = random.Random(seed)
    m = [sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n)]
    m.sort()
    return m[int(.025 * n)], m[int(.975 * n)]


def perm_gap(xa, xb, n=10000, seed=1):
    obs = sum(xa) / len(xa) - sum(xb) / len(xb)
    pool, na, rng, k = xa + xb, len(xa), random.Random(seed), 0
    for _ in range(n):
        rng.shuffle(pool)
        if sum(pool[:na]) / na - sum(pool[na:]) / len(xb) >= obs:
            k += 1
    return obs, (k + 1) / (n + 1)


print(f"parser={WHICH} rows={len(rows)} parse_failures={dict(fail) or 0} judges={judges}")
res = {}
for mode in ["mean", "and"] + judges:
    print(f"\n== unit score: {mode}")
    for arm in sorted(units):
        for idx, name in [(0, "any-type one-job"), (1, "non-lexical one-job")]:
            xs = [score(u, idx, mode) for u in units[arm]]
            xs = [x for x in xs if x is not None]
            lo, hi = boot(xs)
            res[(mode, arm, idx)] = xs
            print(f"  {arm} n={len(xs):3d} {name:20s} {sum(xs)/len(xs):.3f}  CI[{lo:.3f},{hi:.3f}]")
    for arm in sorted(units):  # per layer, descriptive
        line = []
        for L in sorted({meta[u][1] for u in units[arm]}):
            xs = [score(u, 0, mode) for u in units[arm] if meta[u][1] == L]
            xs = [x for x in xs if x is not None]
            line.append(f"L{L}={sum(xs)/len(xs):.2f}")
        print(f"  {arm} any-type by layer: " + " ".join(line))

# type split of one-job verdicts per arm (judge-level)
print("\n== type of one-job verdicts (judge-level counts)")
for arm in sorted(units):
    c = defaultdict(int)
    for u in units[arm]:
        for j in judges:
            if (u, j) in v and v[(u, j)][0]:
                c[v[(u, j)][2]] += 1
    print(" ", arm, dict(c))

# kappa between judges on any-type verdict
if len(judges) == 2:
    for arm in sorted(units) + ["ALL"]:
        us = [u for a in units for u in units[a]] if arm == "ALL" else units[arm]
        pairs = [(v[(u, judges[0])][0], v[(u, judges[1])][0]) for u in us if (u, judges[0]) in v and (u, judges[1]) in v]
        n = len(pairs)
        po = sum(a == b for a, b in pairs) / n
        p1, p2 = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        print(f"  kappa {arm}: {(po - pe) / (1 - pe) if pe < 1 else float('nan'):.3f} (agree {po:.3f}, n={n})")

print("\n== pre-registered hypotheses (unit score = mean of judges)")
arms = sorted(units)
if "A" in arms and "B" in arms:
    obs, p = perm_gap(res[("mean", "A", 1)], res[("mean", "B", 1)])
    print(f"  H1 nonlex A-B = {obs*100:.1f} pp (need >=25), one-sided perm p={p:.4f} -> {'HOLDS' if obs >= .25 else 'FAILS'}")
if "A" in arms and "C" in arms:
    obs, p = perm_gap(res[("mean", "A", 1)], res[("mean", "C", 1)])
    print(f"  H2 nonlex A-C = {obs*100:.1f} pp (need >=15), one-sided perm p={p:.4f} -> {'HOLDS' if obs >= .15 else 'FAILS'}")
if "B" in arms:
    b_any = sum(res[("mean", "B", 0)]) / len(res[("mean", "B", 0)])
    b_lex = b_any - sum(res[("mean", "B", 1)]) / len(res[("mean", "B", 1)])
    print(f"  H3 B any={b_any:.3f} (need >=0.40), lexical part={b_lex:.3f} (need > half of any) -> "
          f"{'HOLDS' if b_any >= .4 and b_lex > b_any / 2 else 'FAILS'}")
if "A" in arms:
    a_any = sum(res[("mean", "A", 0)]) / len(res[("mean", "A", 0)])
    print(f"  H4 A any={a_any:.3f} (need in [0.50,0.90]) -> {'HOLDS' if .5 <= a_any <= .9 else 'FAILS'}")
