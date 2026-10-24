"""Hugo (skeptic2): pair costs on the ellipsoid (D5) from out/ellip_*.json, next to the second-order
pair formula from out/gate_w*.jsonl for the same pairs.
c(a) = min exact KL over found points with shift >= a (shift 0 costs 0).
pair_found = min over a in {0, D/4, D/2, 3D/4, D} of c_false(a) + c_true(D - a), D = s_false - line_clean.
Also a 41-split version (sensitivity). Only pairs whose false statement is caught on the clean model.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ['unproj', 'proj_k128', 'rand_k128']


def cost(sh, kl, a):
    if a <= 0:
        return 0.0
    ok = sh >= a
    return float(kl[ok].min()) if ok.any() else np.inf


gate = {}
for f in glob.glob(os.path.join(HERE, 'out', 'gate_w*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        gate[r['idx']] = r

rows = {a: [] for a in ARMS}
for f in sorted(glob.glob(os.path.join(HERE, 'out', 'ellip_unproj_proj_k128_rand_k128_off*_n*.json'))):
    o = json.load(open(f))
    idx = o['idx']
    for j, a in enumerate(ARMS):
        A = o['arms'][a]
        sh, kl, s0 = np.array(A['shifts']), np.array(A['kls']), np.array(A['s0'])
        for p in range(0, len(idx), 2):
            fi, ti = idx[p], idx[p + 1]
            D = s0[p] - A['line']
            if D <= 0:
                continue
            c5 = min(cost(sh[p], kl[p], x * D) + cost(sh[p + 1], kl[p + 1], (1 - x) * D) for x in np.linspace(0, 1, 5))
            c41 = min(cost(sh[p], kl[p], x * D) + cost(sh[p + 1], kl[p + 1], (1 - x) * D) for x in np.linspace(0, 1, 41))
            cf = cost(sh[p], kl[p], D)
            q = None
            if fi in gate and ti in gate:
                gf, gt = gate[fi]['g']['1e-08'][j], gate[ti]['g']['1e-08'][j]
                q = D ** 2 / (2 * (gf + gt))
            rows[a].append(dict(pair=(fi, ti), D=D, found5=c5, found41=c41, found_false_only=cf, quad_pair=q))

out = {}
for a in ARMS:
    R = rows[a]
    f5 = np.array([r['found5'] for r in R])
    q = np.array([r['quad_pair'] for r in R if r['quad_pair'] is not None])
    out[a] = dict(n=len(R), found5_median=float(np.median(f5)), found5_mean=float(np.mean(f5)),
                  found41_median=float(np.median([r['found41'] for r in R])),
                  quad_median=float(np.median(q)) if len(q) else None,
                  found_over_quad_median=float(np.median([r['found5'] / r['quad_pair'] for r in R if r['quad_pair']])) if len(q) else None,
                  rows=R)
U = {r['pair']: r for r in rows['unproj']}
for a in ARMS:
    rr = [r['found5'] / U[r['pair']]['found5'] for r in rows[a] if r['pair'] in U and np.isfinite(r['found5'])]
    out[a]['per_pair_ratio_vs_U_median'] = float(np.median(rr)) if rr else None
    out[a]['per_pair_ratio_vs_U_range'] = [float(min(rr)), float(max(rr))] if rr else None
    out[a]['ratio_of_means_vs_U'] = out[a]['found5_mean'] / out['unproj']['found5_mean']
json.dump(out, open(os.path.join(HERE, 'out', 'ellip_report.json'), 'w'), indent=1, default=str)
for a in ARMS:
    print(a, {k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in out[a].items() if k != 'rows'})
