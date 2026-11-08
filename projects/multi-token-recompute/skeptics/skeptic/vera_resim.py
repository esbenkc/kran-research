"""Vera: re-sim of the frozen status rule (YES CI-lo>=3 2/2, NO CI-hi<3 2/2) at measured f, s, for R CRN pairs.
usage: vera_resim.py f s [n=200] -> prints P(CONF/REJ/UNC) for true text gaps 0, 1, 2, 3, 5, 10 points at R 4/8/16.
Model as vera_sim.py: unflipped pair difference exactly 0; flipped pair = two independent cheat draws (sd s)
plus g/f. Bootstrap over prompts (B=1000), 400 runs per cell.
"""
import sys, json, numpy as np

f, s = float(sys.argv[1]), float(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rng = np.random.default_rng(11081)
Bt, RUNS = 1000, 400


def ci(per):
    idx = rng.integers(0, len(per), (Bt, len(per)))
    return np.percentile(per[idx].mean(1), [2.5, 97.5])


out = {}
for R in (4, 8, 16):
    for g in (0, 1, 2, 3, 5, 10):
        if g / 100 > f: continue
        c = dict(CONFIRMED=0, REJECTED=0, UNCONFIRMED=0)
        for _ in range(RUNS):
            reads = []
            for _seed in range(2):
                flip = rng.random((n, R)) < f
                d = np.where(flip, rng.normal(0, s, (n, R)) - rng.normal(0, s, (n, R)) + g / 100 / f, 0.0)
                lo, hi = ci(100 * d.mean(1))
                reads.append('YES' if lo >= 3 else 'NO' if hi < 3 else 'MID')
            k = 'CONFIRMED' if reads == ['YES'] * 2 else 'REJECTED' if reads == ['NO'] * 2 else 'UNCONFIRMED'
            c[k] += 1
        out[f'R{R}_g{g}'] = {k: v / RUNS for k, v in c.items()}
        print(R, g, out[f'R{R}_g{g}'], flush=True)
json.dump(dict(f=f, s=s, n=n, cells=out), open(f'vera_resim_f{f}_s{s}.json', 'w'), indent=1)
