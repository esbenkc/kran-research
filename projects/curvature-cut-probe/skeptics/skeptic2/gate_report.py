"""Hugo (skeptic2): power-gate report from out/gate_w*.jsonl (own Fisher) and own probe scores.

Per caught clean false eval statement i (score above its probe's clean line = 2nd-highest of the 200
refit-split true statements), D_i = s_i - line.
  false-only floor_i = D_i^2 / (2 g_false)                (PREREG s6)
  pair floor_i       = D_i^2 / (2 (g_false + g_true))     (D5 second-order pair formula)
Aggregate (D4): sum of the cheapest (n_caught - 119) floors / 1192, estimated from the subsample by
scaling (subsample share of cheapest -> same share; sum scaled by n_caught / n_sub_caught).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
ARMS = ['unproj', 'proj_k128', 'rand_k128', 'lit_k128']


def load():
    recs = {}
    for f in glob.glob(os.path.join(HERE, 'out', 'gate_w*.jsonl')):
        for l in open(f):
            r = json.loads(l)
            recs[r['idx']] = r
    return recs


def scores():
    split = json.load(open(os.path.join(DATA, 'ref/split.json')))
    z = np.load(os.path.join(DATA, 'clean/z_post.npy')).astype(np.float64)
    pz = np.load(os.path.join(DATA, 'fits/probes.npz'))
    tr = np.array([x['true'] for x in split])
    sp = np.array([x['split'] for x in split])
    out = {}
    for a in ARMS:
        w, mu, sd = [pz[f'{a}__{k}'].astype(np.float64) for k in ('w', 'mu', 'sd')]
        s = ((z - mu) / sd) @ w + float(pz[f'{a}__b'])
        line = np.sort(s[(sp == 'refit') & (tr == 1)])[::-1][1]
        out[a] = (s, line)
    return split, tr, sp, out


def agg(fl, n_caught_total, n_sub):
    """D4 aggregate from a subsample of caught floors."""
    need = n_caught_total - 119
    if need <= 0:
        return 0.0
    k = int(round(len(fl) * need / n_caught_total))
    return np.sort(fl)[:k].sum() * (n_caught_total / n_sub) / 1192


def main(eps='1e-08'):
    recs = load()
    split, tr, sp, sc = scores()
    key = {}
    for i, x in enumerate(split):
        if x['split'] == 'eval':
            key.setdefault((x['city'], x['src']), {})[x['true']] = i
    pairs = [(v[0], v[1]) for v in key.values() if v[0] in recs and v[1] in recs]
    res = {'n_pairs_done': len(pairs), 'eps_rel': eps, 'arms': {}}
    for j, a in enumerate(ARMS):
        s, line = sc[a]
        ev_f = (sp == 'eval') & (tr == 0)
        n_caught = int((s[ev_f] > line).sum())
        D, gf, gt = [], [], []
        for fi, ti in pairs:
            if s[fi] > line:
                D.append(s[fi] - line)
                gf.append(recs[fi]['g'][eps][j])
                gt.append(recs[ti]['g'][eps][j])
        D, gf, gt = map(np.array, (D, gf, gt))
        fo = D ** 2 / (2 * gf)
        fp = D ** 2 / (2 * (gf + gt))
        res['arms'][a] = dict(line=float(line), clean_catch=n_caught / 596, n_caught=n_caught, n_sub=len(D),
                              D_median=float(np.median(D)), falseonly_median=float(np.median(fo)),
                              pair_median=float(np.median(fp)),
                              falseonly_agg=agg(fo, n_caught, len(D)), pair_agg=agg(fp, n_caught, len(D)),
                              _fo=fo, _fp=fp, _pairs=[p for p in pairs if s[p[0]] > line])
    U = res['arms']['unproj']
    for a in ARMS:
        r = res['arms'][a]
        r['ratio_pair_agg_vs_U'] = r['pair_agg'] / U['pair_agg']
        r['ratio_falseonly_agg_vs_U'] = r['falseonly_agg'] / U['falseonly_agg']
        # per-pair ratio on pairs caught by both
        common = [p for p in r['_pairs'] if p in U['_pairs']]
        ia = {p: k for k, p in enumerate(r['_pairs'])}
        iu = {p: k for k, p in enumerate(U['_pairs'])}
        rr = np.array([r['_fp'][ia[p]] / U['_fp'][iu[p]] for p in common])
        r['per_pair_ratio_median'] = float(np.median(rr)) if len(rr) else None
        r['per_pair_ratio_iqr'] = [float(np.percentile(rr, 25)), float(np.percentile(rr, 75))] if len(rr) else None
    for a in ARMS:
        for k in ('_fo', '_fp', '_pairs'):
            res['arms'][a].pop(k)
    return res


if __name__ == '__main__':
    out = {e: main(e) for e in ('1e-10', '1e-08', '1e-06')}
    json.dump(out, open(os.path.join(HERE, 'out', 'gate_report.json'), 'w'), indent=1)
    r = out['1e-08']
    print('pairs done', r['n_pairs_done'])
    for a, v in r['arms'].items():
        print(a, {k: (round(x, 6) if isinstance(x, float) else x) for k, x in v.items()})
    for e in ('1e-10', '1e-06'):
        print('eps', e, {a: round(v['ratio_pair_agg_vs_U'], 3) for a, v in out[e]['arms'].items()})
