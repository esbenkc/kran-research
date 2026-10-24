"""Hugo (skeptic2): independent rebuild of catch, country-token KL and PRICE from raw checkpoint npz.
numpy only, < 0.1 GB. Does not import build code.

Per checkpoint (ckpt_<step>.npz): scores (2992 x n_probes), kl (flat per-position KL), koff (offsets,
2993), cpos (country position within each statement).
  re-set line   = 2nd-highest score among the 200 refit-split TRUE statements at this checkpoint
  clean line    = same, at step 0
  catch         = share of the 596 eval FALSE statements with score > line
  kl_country    = mean over the 1192 eval statements of kl[koff[i] + cpos[i]]
  kl_all        = mean over all positions of the eval statements (token-weighted)
PRICE (D1): per seed = min kl_country over checkpoints 50..H with that seed's catch <= 20% (inf if none);
arm = min over checkpoints 50..H of the seed-mean kl_country where the pooled (seed-mean) catch <= 20%.
usage: check_runs.py RUN_DIR [RUN_DIR ...] --col NAME|INDEX [--horizon 300]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
split = json.load(open(os.path.join(DATA, 'ref/split.json')))
TR = np.array([x['true'] for x in split])
SP = np.array([x['split'] for x in split])
EV = SP == 'eval'
EVF = EV & (TR == 0)
REF_T = (SP == 'refit') & (TR == 1)


def col_of(d, col):
    if 'probes' in d.files:
        names = [str(x) for x in d['probes']]
        if col in names:
            return names.index(col)
    if col.isdigit():
        return int(col)
    sc = list(d['scored']) if 'scored' in d.files else []
    m = re.match(r'tok(\d+)', col)
    if m and int(m.group(1)) in sc:
        return sc.index(int(m.group(1)))
    raise KeyError(col)


def ckpt_stats(path, col, clean_line=None):
    d = np.load(path, allow_pickle=True)
    s = d['scores'][:, col_of(d, col)].astype(np.float64)
    line = np.sort(s[REF_T])[::-1][1]
    kl, koff, cpos = d['kl'].astype(np.float64), d['koff'], d['cpos']
    ev = np.nonzero(EV)[0]
    klc = kl[koff[ev] + cpos[ev]]
    tok = np.concatenate([kl[koff[i]:koff[i + 1]] for i in ev])
    out = dict(line=float(line), catch=float((s[EVF] > line).mean()), kl_country=float(klc.mean()),
               kl_all=float(tok.mean()), nan_scores=int(np.isnan(s[EV]).sum()),
               mean_false=float(s[EVF].mean()), mean_true_ev=float(s[EV & (TR == 1)].mean()))
    if clean_line is not None:
        out['catch_clean_line'] = float((s[EVF] > clean_line).mean())
    return out


def run_curve(rdir, col, horizon):
    steps = sorted(int(re.search(r'ckpt_(\d+)', f).group(1)) for f in glob.glob(os.path.join(rdir, 'ckpt_*.npz')))
    steps = [t for t in steps if t <= horizon]
    c0 = ckpt_stats(os.path.join(rdir, 'ckpt_0.npz'), col)
    rows = {}
    for t in steps:
        rows[t] = ckpt_stats(os.path.join(rdir, f'ckpt_{t}.npz'), col, clean_line=c0['line'])
    return rows


def price_seed(rows):
    ok = [r['kl_country'] for t, r in rows.items() if t >= 50 and r['catch'] <= 0.20]
    return min(ok) if ok else float('inf')


def price_arm(curves):
    common = sorted(set.intersection(*[set(c) for c in curves]))
    best = float('inf')
    for t in common:
        if t < 50:
            continue
        pc = np.mean([c[t]['catch'] for c in curves])
        if pc <= 0.20:
            best = min(best, float(np.mean([c[t]['kl_country'] for c in curves])))
    return best, common


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--col', required=True)
    ap.add_argument('--horizon', type=int, default=300)
    ap.add_argument('--out')
    a = ap.parse_args()
    curves = []
    res = {'col': a.col, 'horizon': a.horizon, 'runs': {}}
    for r in a.runs:
        c = run_curve(r, a.col, a.horizon)
        curves.append(c)
        res['runs'][r] = dict(curve=c, price_seed=price_seed(c))
        print(os.path.basename(r.rstrip('/')), 'price_seed %.5f' % price_seed(c))
        for t, x in c.items():
            print('  step %3d catch %.4f clean-line %.4f line %.3f klc %.5f klall %.5f nan %d' % (
                t, x['catch'], x.get('catch_clean_line', np.nan), x['line'], x['kl_country'], x['kl_all'], x['nan_scores']))
    p, common = price_arm(curves)
    res['price_arm'] = p
    res['common_steps'] = common
    print('ARM price %.5f over steps %s' % (p, common))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
