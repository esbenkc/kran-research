"""critic2 (Claude fallback) rebuild: switch times, per-seed slopes, seed-level and Welch intervals,
S2, C1 and G2 from the raw run JSONs. Own code; uses only json + numpy + scipy.stats.t."""
import json, glob, os, math
import numpy as np
from scipy import stats
H = os.path.join(os.path.dirname(__file__), '..', '..')
def evals(d):
    e = {r['step']: r for r in d['dev_curve'] + d['dense_curve']}
    g0 = e[0]['px_off'] - e[0]['px_on']
    st = sorted(e)
    return [(s, (e[s]['px_off'] - e[s]['px_on']) - g0) for s in st], g0
def switch(d):
    ev, g0 = evals(d)
    for (s, D), (s2, D2) in zip(ev, ev[1:]):
        if s > 0 and D >= 0.10 and D2 >= 0.10:
            return s, g0, max(x for _, x in ev)
    return None, g0, max(x for _, x in ev)
def arm(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(H, pat))):
        d = json.load(open(f))
        if d['args']['arm'] != 'cue': continue
        T, g0, mx = switch(d)
        out[(d['args']['seed'], d['k'])] = dict(T=T, share=d['share'], g0=g0, Tstore=d['T_dense'])
    return out
def slopes(a, seeds, ks=range(6)):
    b = {}
    for s in seeds:
        pts = [(a[(s, k)]['share'], a[(s, k)]['T']) for k in ks if (s, k) in a and a[(s, k)]['T'] and abs(a[(s, k)]['g0']) < 0.02]
        if len(pts) < 4 or len(pts) < len(list(ks)): continue
        x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        b[s] = np.polyfit(x, y, 1)[0]
    return b
def tint(v):
    v = np.array(list(v)); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h
r = {}
a8 = arm('data/runs/m8_k*_s*.json'); a6 = arm('data/runs_06b/m06_k*_s*.json')
r['T8'] = {f'{s}/{k}': v['T'] for (s, k), v in sorted(a8.items())}
r['T6'] = {f'{s}/{k}': v['T'] for (s, k), v in sorted(a6.items())}
r['stored_mismatch'] = [f'{key}' for A in (a8, a6) for key, v in A.items() if v['T'] != v['Tstore']]
b8 = slopes(a8, [0, 1, 2]); b6 = slopes(a6, [0, 1, 2]); b6all = slopes(a6, [0, 1, 2, 3, 4], ks=range(6))
r['b8'] = b8; r['b8_int'] = tint(b8.values()); r['b6'] = b6; r['b6_int'] = tint(b6.values())
# Welch on slope diff
v8, v6 = np.array(list(b8.values())), np.array(list(b6.values()))
s8, s6 = v8.var(ddof=1) / len(v8), v6.var(ddof=1) / len(v6)
dfw = (s8 + s6) ** 2 / (s8 ** 2 / (len(v8) - 1) + s6 ** 2 / (len(v6) - 1))
dd = v8.mean() - v6.mean(); hw = stats.t.ppf(0.975, dfw) * math.sqrt(s8 + s6)
r['S1'] = (dd, dd - hw, dd + hw, dfw)
r['per_halving_8b'] = [2 ** -x for x in r['b8_int']]
r['per_halving_06b'] = 2 ** -r['b6_int'][0]
med8 = [float(np.median([a8[(s, k)]['T'] for s in range(3)])) for k in range(6)]
med6 = [float(np.median([a6[(s, k)]['T'] for s in range(3)])) for k in range(6)]
r['med8'] = med8; r['med6'] = med6; r['ratio8_6'] = [m8 / m6 for m8, m6 in zip(med8, med6)]
pred = [med8[0] * 2 ** (0.90265 * k) for k in range(6)]
r['S2_pred'] = pred; r['S2_ratio'] = [m / p for m, p in zip(med8, pred)]
r['halving_ratios8'] = [med8[k + 1] / med8[k] for k in range(5)]
# pooled OLS across 18 8B cells
xs = [math.log(a8[(s, k)]['share']) for s in range(3) for k in range(6)]; ys = [math.log(a8[(s, k)]['T']) for s in range(3) for k in range(6)]
r['pooled8'] = float(np.polyfit(xs, ys, 1)[0])
c = json.load(open(os.path.join(H, 'data/runs/m8_c1_s0.json')))
Tc, gc, mxc = switch(c)
r['C1s0'] = dict(T=Tc, maxD=mxc, last_step=max(e['step'] for e in c['dev_curve'] + c['dense_curve']), resumed=c.get('resumed_from'))
g = json.load(open(os.path.join(H, 'data/gates.json')))
pp = g['G2']['per_prompt']
dl = [k for k in pp if 'dln' in k.lower() or 'lnp' in k.lower()]
r['G2_keys'] = list(pp.keys())
r['G2'] = {k: g['G2'][k] for k in ('a_rel_err_max', 'a_pass', 'b_abs_dlnpx_max', 'b_pass', 'b_top1_agree')}
for k in dl:
    v = np.abs(np.array(pp[k], float)); r['G2_' + k] = dict(max=float(v.max()), n_gt_005=int((v > 0.05).sum()), n=len(v))
r['G1'] = {k: v for k, v in g.items() if k.startswith('G1')}
print(json.dumps(r, indent=1, default=float))
json.dump(r, open(os.path.join(os.path.dirname(__file__), 'rebuild.json'), 'w'), indent=1, default=float)
# --- secondary reads: pooled OLS interval, two halves, grid T (S5)
from scipy import stats as st2
x = np.array(xs); y = np.array(ys); lr = st2.linregress(x, y)
h = st2.t.ppf(0.975, len(x) - 2) * lr.stderr
r2 = {'pooled_ci': (lr.slope, lr.slope - h, lr.slope + h)}
def fit(ks, key='T', pat='data/runs/m8_k*_s*.json'):
    out = []
    ds = [json.load(open(f)) for f in glob.glob(os.path.join(H, pat))]
    for s in range(3):
        pts = [(d['share'], d[key]) for d in ds if d['args']['arm'] == 'cue' and d['args']['seed'] == s and d['k'] in ks]
        out.append(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
    return tint(out)
r2['halves'] = (fit([0, 1, 2], 'T_dense'), fit([3, 4, 5], 'T_dense'))
r2['S5_grid'] = fit(range(6), 'T')
print(json.dumps(r2, default=float))
r.update(r2); json.dump(r, open(os.path.join(os.path.dirname(__file__), 'rebuild.json'), 'w'), indent=1, default=float)
