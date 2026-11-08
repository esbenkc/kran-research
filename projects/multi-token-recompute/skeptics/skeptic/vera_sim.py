"""Vera: power of the frozen k16 status read under common random numbers (CRN) + Rao-Blackwellised cheat.

Per seed: n test prompts x m CRN pairs. A pair's text-part difference is EXACTLY 0 when the probe flipped no
prefix token (same text -> same clean answer pass). With prob f it flipped; then
D = q(off prefix) - q(on prefix) + g_true / f, with q the answer-slot cheat prob and the two q's independent
draws around the prompt mean with sd s (clipped to [0,1] in effect by the small s used).
Gap = mean over prompts of the per-prompt mean D, CI = percentile bootstrap over prompts (B=2000).
Status per seed: YES10 = gap >= 10 and CI-lo > 0 (Felix draft); YES3 = CI-lo >= 3 (Vera draft);
NO = CI-hi < 3. Title status needs 2/2 seeds on the same side (planted gate assumed passed here).
"""
import json, itertools, numpy as np

rng = np.random.default_rng(1108)
B = 2000


def seed_read(n, m, f, s, g):
    flip = rng.random((n, m)) < f
    d = (rng.normal(0, s, (n, m)) - rng.normal(0, s, (n, m)))
    d = np.where(flip, d + (g / max(f, 1e-9)), 0.0)
    per = d.mean(1) * 100  # points
    gap = per.mean()
    idx = rng.integers(0, n, (B, n))
    bs = per[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return gap, lo, hi


def status(reads, rule):
    def one(r):
        gap, lo, hi = r
        if rule == 'felix':
            if gap >= 10 and lo > 0: return 'YES'
        else:
            if lo >= 3: return 'YES'
        if hi < 3: return 'NO'
        return 'MID'
    a = [one(r) for r in reads]
    if all(x == 'YES' for x in a): return 'CONFIRMED'
    if all(x == 'NO' for x in a): return 'REJECTED'
    return 'UNCONFIRMED'


out = []
R = 400
for n, m, f, s, g in itertools.product([200, 400], [1, 4], [0.02, 0.05, 0.1, 0.2, 0.4], [0.1, 0.2, 0.3],
                                       [0.0, 0.01, 0.02, 0.05, 0.10]):
    if g > f: continue  # text gap cannot exceed the flip share under CRN
    for rule in ['felix', 'vera']:
        c = {'CONFIRMED': 0, 'REJECTED': 0, 'UNCONFIRMED': 0}
        for _ in range(R):
            c[status([seed_read(n, m, f, s, g) for _ in range(2)], rule)] += 1
        out.append(dict(n=n, m=m, f=f, s=s, g_true_pts=g * 100, rule=rule, **{k: v / R for k, v in c.items()}))
json.dump(out, open('vera_sim.json', 'w'), indent=1)
for o in out:
    if o['rule'] == 'vera' or o['g_true_pts'] in (0, 5, 10):
        print(o)
