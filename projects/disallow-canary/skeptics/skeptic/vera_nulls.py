"""Null tests for disallow-canary (Vera). Reads vera_check's own parse, no build code.
N1 code-shuffle: score each main run against ANOTHER run's canary code (random derangement within model).
   A clean attribution gives 0 catches; any hit means codes leak across runs or collide.
N2 arm-label permutation within run index i (main vs bare, paired on i where both exist): two-sided p for
   the observed main-bare catch difference, 20,000 swaps. Tests the attribution sentence, not the status.
N3 block-label shuffle: is the seed-level spread larger than binomial noise? (shuffle runs across blocks)
Usage: python vera_nulls.py <project dir>
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import vera_check as vc

P = Path(sys.argv[1])
vc.D = P / "data"
rng = random.Random(1128)

log = vc.parse_log(vc.D / "logs" / "access.log")
by_code = defaultdict(list)
import re
for r in log:
    for m in re.finditer(r"/r/([0-9a-f]{12})", r["path"]):
        by_code[m.group(1)].append(r)

catch = {}
codes = {}
for model in vc.MODELS:
    for arm in vc.ARMS:
        for f in sorted((vc.D / "runs" / model / arm).glob("*.json")):
            i = int(f.stem)
            st = vc.site_truth(f"{model}-{arm}-{i:03d}")
            codes[(model, arm, i)] = st["code"]
            catch[(model, arm, i)] = len(by_code.get(st["code"], [])) > 0

print("N1 code-shuffle (catches when scored with another run's code; want 0):")
for model in vc.MODELS:
    for arm in vc.ARMS:
        keys = [k for k in codes if k[0] == model and k[1] == arm]
        tot = 0
        for _ in range(20):
            perm = keys[:]
            while any(a == b for a, b in zip(keys, perm)) and len(keys) > 1:
                rng.shuffle(perm)
            tot += sum(any(h["tag"] == f"{k[0]}/{k[1]}/{k[2]:03d}" for h in by_code.get(codes[p], []))
                       for k, p in zip(keys, perm))
        print(f"  {model}/{arm}: {tot} cross-attributed catches over 20 derangements of {len(keys)} runs")

print("N2 arm-label permutation within i (main vs bare):")
for model in vc.MODELS:
    pairs = [(catch[(model, 'main', i)], catch[(model, 'bare', i)]) for (m, a, i) in codes if m == model and a == 'bare' and (model, 'main', i) in catch]
    n = len(pairs)
    if not n:
        continue
    obs = sum(a - b for a, b in pairs) / n
    ge = 0
    R = 20000
    for _ in range(R):
        s = 0
        for a, b in pairs:
            if rng.random() < 0.5:
                a, b = b, a
            s += a - b
        ge += abs(s / n) >= abs(obs) - 1e-12
    print(f"  {model}: paired n={n}, main-bare = {obs:+.3f}, two-sided permutation p = {ge / R:.4f}")

print("N3 block shuffle (seed-level sd vs runs reshuffled across blocks):")
from statistics import pstdev
for model in vc.MODELS:
    runs = sorted((i, catch[(model, 'main', i)]) for (m, a, i) in codes if m == model and a == 'main')
    blocks = defaultdict(list)
    for i, c in runs:
        blocks[i % 3].append(c)
    sd_obs = pstdev([sum(v) / len(v) for v in blocks.values()])
    vals = [c for _, c in runs]
    sizes = [len(v) for v in blocks.values()]
    ge = 0
    R = 5000
    for _ in range(R):
        rng.shuffle(vals)
        o = 0
        sh = []
        for s in sizes:
            sh.append(sum(vals[o:o + s]) / s)
            o += s
        ge += pstdev(sh) >= sd_obs - 1e-12
    print(f"  {model}: block shares {[round(sum(v)/len(v),3) for k,v in sorted(blocks.items())]}, sd {sd_obs:.3f}, P(shuffled sd >= obs) = {ge / R:.3f}")
