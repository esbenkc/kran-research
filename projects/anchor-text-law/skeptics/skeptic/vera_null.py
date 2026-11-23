# Vera null tests on the raw rows (uses vera_check.py's own loaders; no build code).
# (1) format-label shuffle within a model: pool bare+md runs, reassign labels at random, recompute fr(bare)-fr(md).
#     How often does G (gap >= 15 pts) fire by luck? Same for anchor vs md (|diff| <= 10 by luck).
# (2) law-vs-null: shuffle the law's 'format class' (which fit cell feeds follow_A: raw vs bare) -> the format-blind
#     null error in the A status cells.
# (3) seed-level t-interval null: shuffle bare/md labels, recompute the seed-level interval of fr(bare)-fr(md);
#     how often does it exclude 0 by luck (nominal 5%)?
import sys, random, math
sys.argv = [sys.argv[0]] + sys.argv[1:]
import vera_check as V
from fractions import Fraction as F

a, b = V.model_dirs()
evA, _, _ = V.build(a[0]); evB, _, _ = V.build(b[0])
nA, nB = V.scored_n(evA), V.scored_n(evB)
rng = random.Random(20261123)
S = 20000


def fr(es):
    r = [e for e in es if e['reach']]
    return sum(e['follow'] for e in r) / len(r) if r else float('nan')


for m, ev, n in (('A', evA, nA), ('B', evB, nB)):
    for other in ('bare', 'anchor'):
        x = [ev[other][i] for i in range(n)]; y = [ev['md'][i] for i in range(n)]
        obs = fr(x) - fr(y)
        pool = x + y
        g = 0; c10 = 0; more = 0
        seedx = 0
        for _ in range(S):
            rng.shuffle(pool)
            d = fr(pool[:n]) - fr(pool[n:])
            g += d <= -0.15; c10 += abs(d) <= 0.10; more += abs(d) >= abs(obs) - 1e-12
        print(f'{m} fr({other})-fr(md) observed {obs:+.4f}; shuffled: P(diff <= -0.15) {g/S:.4f}, '
              f'P(|diff| <= 0.10) {c10/S:.4f}, two-sided p(|diff| >= |obs|) {more/S:.4f}  (n {n} each, {S} shuffles)')
        # seed-level interval null
        excl = 0; T = 4000
        for _ in range(T):
            idx = list(range(2 * n)); rng.shuffle(idx)
            lab = {j: (j < n) for j in idx[:0]}
            px = [pool[j] for j in idx[:n]]; py = [pool[j] for j in idx[n:]]
            vals = []
            for bb in range(3):
                vals.append(fr(px[bb::3]) - fr(py[bb::3]))
            if any(math.isnan(v) for v in vals):
                continue
            mu, lo, hi = V.tint(vals) if len(set(vals)) > 1 else (vals[0], vals[0], vals[0])
            excl += (lo > 0 or hi < 0)
        print(f'   seed-level t-interval of shuffled diff excludes 0 in {excl/T:.4f} of {T} shuffles (nominal 0.05)')
