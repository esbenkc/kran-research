# T5 nulls (Vera), on rebuild_rows.json written by vera_rebuild.py.
# (a) paired BEFORE vs AFTER per harness: same run index i = same site/seed (common random numbers) -> discordant pairs,
#     exact McNemar p; (b) label-swap permutation null: swap BEFORE/AFTER labels within each pair at random,
#     20,000 draws, two-sided p for the pooled AFTER-minus-BEFORE gap; (c) per seed block, per model, per template
#     direction of the gap; (d) NONE vs AFTER paired (presence-vs-order: is AFTER ~ NONE?).
import json, pathlib, random, math, collections, sys
rows = json.load(open(pathlib.Path(__file__).with_name("rebuild_rows.json")))
rng = random.Random(1126)
def binom_two(k, n):
    if n == 0: return 1.0
    pk = sum(math.comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * pk)
for h in ("web", "shell"):
    pairs = []; pna = []
    by = collections.defaultdict(dict)
    for m, rr in rows.items():
        for r in rr:
            if r["harness"] == h and r["exposed"]:
                by[(m, r["i"])][r["arm"]] = r
    for (m, i), d in by.items():
        if "before" in d and "after" in d: pairs.append((m, i, d["before"]["trap"], d["after"]["trap"]))
        if "none" in d and "after" in d: pna.append((m, i, d["none"]["trap"], d["after"]["trap"]))
    if not pairs: continue
    b10 = sum(1 for *_, b, a in pairs if b and not a); b01 = sum(1 for *_, b, a in pairs if a and not b)
    gap = (sum(a for *_, b, a in pairs) - sum(b for *_, b, a in pairs)) / len(pairs)
    cnt = 0; N = 20000
    for _ in range(N):
        s = 0
        for *_, b, a in pairs:
            if rng.random() < 0.5: b, a = a, b
            s += a - b
        if abs(s / len(pairs)) >= abs(gap) - 1e-12: cnt += 1
    print(f"== {h}: {len(pairs)} paired runs; AFTER-minus-BEFORE gap {gap:+.3f}; discordant before-only {b10}, after-only {b01}; "
          f"McNemar exact p {binom_two(min(b10, b01), b10 + b01):.4g}; label-swap permutation p {cnt / N:.4g}")
    n10 = sum(1 for *_, x, a in pna if x and not a); n01 = sum(1 for *_, x, a in pna if a and not x)
    print(f"   NONE vs AFTER paired: {len(pna)} pairs; none-only {n10}, after-only {n01}; McNemar p {binom_two(min(n10, n01), n10 + n01):.4g}")
    for key in ("block", "model"):
        g = collections.defaultdict(lambda: [0, 0, 0])
        for m, i, b, a in pairs:
            k = i % 3 if key == "block" else m
            g[k][0] += b; g[k][1] += a; g[k][2] += 1
        print(f"   by {key}: " + "; ".join(f"{k}: before {v[0]}/{v[2]}, after {v[1]}/{v[2]}" for k, v in sorted(g.items())))
