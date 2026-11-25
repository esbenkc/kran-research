"""Vera: how close is the landed result to another status? Uses vera_check.status (the frozen rule) only.

E1 one-count edges: smallest single-arm count change that flips the status, per arm.
E2 parametric bootstrap at the observed per-arm rates (n = 90): P(CONFIRMED / REJECTED / UNCONFIRMED / gap fail).
E3 the 'nofollow does nothing' truth fitted to this run (pool raw+rawnf, pool mdhtml+mdnf; md as observed):
   P(REJECTED) = how often the frozen rule could say No if the attribute truly does nothing here.
E4 'syntax only' truth fitted: same as E3 (pooled pairs) - identical by construction; reported once.
Usage: python3 vera_edges.py [draws]
"""
import sys, random, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import vera_check as v

obs = {'raw': 72, 'rawnf': 74, 'md': 82, 'mdhtml': 74, 'mdnf': 70}
N = {a: 90 for a in obs}
S = int(sys.argv[1]) if len(sys.argv) > 1 else 50000

s0 = v.status(obs, N)[0]
print('observed', obs, '->', s0)

print('E1 one-arm edges (nearest count change up and down that flips the status):')
seen = {}
for a in obs:
    for d in [x for k in range(1, 20) for x in (k, -k)]:
        c = dict(obs); c[a] += d
        if not 0 <= c[a] <= 90:
            continue
        s = v.status(c, N)[0]
        if s != s0 and not seen.get((a, d > 0)):
            seen[(a, d > 0)] = 1
            print(f'  {a} {obs[a]}{d:+d} -> {s}')

def sim(p, label):
    rng = random.Random(1124)
    t = collections.Counter()
    for _ in range(S):
        c = {a: sum(rng.random() < p[a] for _ in range(90)) for a in p}
        t[v.status(c, N)[0]] += 1
    print(label, {k: round(x / S, 4) for k, x in sorted(t.items())})

sim({a: obs[a] / 90 for a in obs}, 'E2 bootstrap at observed rates:')
pr = (obs['raw'] + obs['rawnf']) / 180
pm = (obs['mdhtml'] + obs['mdnf']) / 180
sim({'raw': pr, 'rawnf': pr, 'md': obs['md'] / 90, 'mdhtml': pm, 'mdnf': pm},
    f'E3 attribute-null fitted (raw=rawnf={pr:.3f}, mdhtml=mdnf={pm:.3f}):')
