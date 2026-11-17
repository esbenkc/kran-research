"""Vera: re-derive the frozen head sets from each audit file's per-head 'removed_dev' field (PREREG FROZEN RULE:
SET = removed >= 0.05, empty -> top-1; TOP-1 = max removed; LOO = all 64 minus the smallest |removed| (ties: lower
block, then lower index); RANDOM = |SET| heads outside SET, random.Random(1117 + seed), draws 1 and 2), and recompute
the t=0 'removed' of each arm from the saved rows: removed = 1 - mean push(clamped) / mean push(unclamped),
push = [logpA - logpB](probe on) - (probe off). Usage: python3 vera_sets.py AUDIT_DIR"""
import json, sys, glob, os, random, collections, numpy as np

def push(rows):
    by = collections.defaultdict(dict)
    for pid, probe, top1, la, lb in rows: by[pid][int(probe)] = la - lb
    return float(np.mean([d[1] - d[0] for d in by.values()]))

def key(h):
    b, i = h.split("."); return (int(b), int(i))

for f in sorted(glob.glob(os.path.join(sys.argv[1], "s*.json"))):
    d = json.load(open(f)); seed = d["seed"]; a = d["audit"]
    rem = {h: v["removed_dev"] for h, v in a.items()}
    heads = sorted(rem, key=key); assert len(heads) == 64, len(heads)
    S = sorted([h for h in heads if rem[h] >= 0.05], key=lambda h: -rem[h]) or [max(heads, key=lambda h: rem[h])]
    top1 = max(heads, key=lambda h: rem[h])
    free = min(heads, key=lambda h: (abs(rem[h]), key(h)))
    outside = [h for h in heads if h not in S]
    rng = random.Random(1117 + seed); draws = [sorted(rng.sample(outside, len(S)), key=key) for _ in range(2)]
    st = d["sets"]
    chk = dict(set=sorted(S, key=key) == sorted(st["set"], key=key), top1=[top1] == st["top1"],
               leave1=sorted(set(heads) - {free}, key=key) == sorted(st["leave1"], key=key) and d["free_head"] == free)
    for k in ("random1", "random2"):
        if k in st: chk[k] = sorted(st[k], key=key) == draws[int(k[-1]) - 1]
    R = d["rows"]; p0 = push(R["noclamp/dev"])
    remt0 = {arm: 1 - push(R[f"t0_{arm}/dev"]) / p0 for arm in ("set", "top1", "leave1", "all64", "random1", "random2") if f"t0_{arm}/dev" in R}
    print(f"s{seed}: SET {sorted(S, key=key)} (removed {[round(rem[h], 3) for h in S]}), next {round(sorted(rem.values())[-len(S)-1], 3)}; "
          f"TOP-1 {top1}; free {free} ({rem[free]:+.4f}); RANDOM draws {draws}")
    print(f"     match stored sets: {chk}; growth {d.get('set_growth')}; t0 removed (dev, from rows): "
          + ", ".join(f"{k} {v:.3f}" for k, v in remt0.items()))
