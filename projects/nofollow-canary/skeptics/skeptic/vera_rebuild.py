"""Vera's independent rebuild of the nofollow-canary counts from raw access logs.

Reads only manifest.jsonl + access.log (+ raw/*.json for reply checks). Own code, no reuse.
Usage: python3 vera_rebuild.py <qwen3-8b out dir> [--null N]
"""
import json, math, random, sys
from collections import defaultdict
from fractions import Fraction

OUT = sys.argv[1]
NNULL = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else 0
Z = 1.959964

rows = {}
for line in open(f"{OUT}/manifest.jsonl"):
    r = json.loads(line)
    rows[(r["cell"], r["i"])] = r  # last row wins (crash reruns)

gets = defaultdict(list)  # run key -> [(method, path, status)]
for line in open(f"{OUT}/access.log"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 7:
        continue
    ts, ip, meth, path, status, ua, run = p[:7]
    gets[run].append((meth, path, int(status), ua))

def followed(r):
    key = f'{r["cell"]}/{r["i"]:03d}'
    return any(m == "GET" and r["hid"] in path for m, path, st, ua in gets[key])

def follow200(r):
    key = f'{r["cell"]}/{r["i"]:03d}'
    return any(m == "GET" and r["hid"] in path and st == 200 for m, path, st, ua in gets[key])

arms = defaultdict(list)
for (cell, i), r in sorted(rows.items()):
    arms[cell.replace("-free", "")].append(r)

def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h

def newcombe(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

def status(c, n):
    F = lambda a: Fraction(c[a], n[a])
    gap = F("md") - F("raw") >= Fraction(1, 10)
    if not gap:
        return "UNCONFIRMED(gap)"
    conf = (newcombe(c["rawnf"], n["rawnf"], c["raw"], n["raw"])[1] > 0
            and newcombe(c["mdhtml"], n["mdhtml"], c["mdnf"], n["mdnf"])[1] > 0
            and F("rawnf") >= F("md") - Fraction(1, 10)
            and F("mdnf") <= F("raw") + Fraction(1, 10))
    if conf:
        return "CONFIRMED"
    ra = newcombe(c["md"], n["md"], c["rawnf"], n["rawnf"])[1] > 0
    rb = (newcombe(c["mdnf"], n["mdnf"], c["raw"], n["raw"])[1] > 0
          or F("mdnf") >= F("mdhtml") - Fraction(1, 10))
    return "REJECTED" if (ra and rb) else "UNCONFIRMED"

c, n = {}, {}
print("arm      n  follow  follow200  catch   wilson95        seeds(blocks by seed%3)")
for a in ["raw", "rawnf", "md", "mdhtml", "mdnf", "mdpad8", "gatemd"]:
    rs = arms[a]
    f = [followed(r) for r in rs]
    f2 = sum(follow200(r) for r in rs)
    c[a], n[a] = sum(f), len(rs)
    blocks = defaultdict(lambda: [0, 0])
    for r, ff in zip(rs, f):
        b = blocks[r["seed"] % 3]; b[0] += ff; b[1] += 1
    lo, hi = wilson(c[a], n[a])
    print(f"{a:7s} {n[a]:3d} {c[a]:6d} {f2:9d}  {c[a]/n[a]:.4f}  [{lo:.3f},{hi:.3f}]  {dict(sorted(blocks.items()))}")

print()
for x, y in [("rawnf", "raw"), ("mdhtml", "mdnf"), ("md", "rawnf"), ("mdnf", "raw"), ("md", "raw"), ("md", "mdhtml")]:
    d, lo, hi = newcombe(c[x], n[x], c[y], n[y])
    print(f"{x}-{y}: {d:+.4f} Newcombe [{lo:+.4f}, {hi:+.4f}]")
print("STATUS (v2 frozen rule, my copy):", status(c, n))

if NNULL:
    # null: shuffle arm labels within each site index i across the 5 status arms (same site/seed design)
    rng = random.Random(20261125)
    by_i = defaultdict(dict)
    for a in ["raw", "rawnf", "md", "mdhtml", "mdnf"]:
        for r in arms[a]:
            by_i[r["i"]][a] = followed(r)
    tally = defaultdict(int)
    for _ in range(NNULL):
        cc = defaultdict(int); nn = defaultdict(int)
        for i, d in by_i.items():
            labs = list(d); vals = [d[k] for k in labs]; rng.shuffle(vals)
            for k, v in zip(labs, vals):
                cc[k] += v; nn[k] += 1
        tally[status(cc, nn)] += 1
    print(f"NULL (labels shuffled within site i, {NNULL} draws):", dict(tally))
