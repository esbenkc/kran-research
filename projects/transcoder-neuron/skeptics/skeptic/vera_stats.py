"""Vera's own statistics rebuild from Bruno's raw CSVs (data/pushes/L{L}_{units,pushes}.csv), per PREREG v2.6.
No teammate code is used. Output: vera_stats.json."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

import os
D = os.environ.get('VD') or '/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron/data/pushes'
rng = np.random.default_rng(12345)
LAYERS = [int(a) for a in sys.argv[1].split(',')] if len(sys.argv) > 1 else [8, 9, 10, 11]
GROUPS = ['NEURON', 'GAP', 'REST', 'ORACLE']


def r2yx(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot_units(df, pc, mc, B=2000):
    units = df['uid'].values
    uq, inv = np.unique(units, return_inverse=True)
    p, y = df[pc].values, df[mc].values
    idx_by = [np.where(inv == k)[0] for k in range(len(uq))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uq), len(uq))
        ii = np.concatenate([idx_by[k] for k in pick])
        out.append(r2yx(p[ii], y[ii]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def verdict(n, r):
    if n >= 0.7 and r <= 0.4: return 'YES'
    if r >= 0.7 or n < 0.4 or n - r < 0.1: return 'NO'
    return 'NEITHER'


def no_clause(n, r):
    c = []
    if r >= 0.7: c.append('both faithful (REST>=0.7)')
    if n < 0.4: c.append('neuron copies fail too (NEURON<0.4)')
    if n - r < 0.1: c.append('no difference (gap<0.1)')
    return c


def relerr(df, pc, mc):
    a = np.abs(df[mc] - df[pc]) / (np.abs(df[mc]) + np.abs(df[pc]) + 1e-12)
    return a.groupby(df['uid']).median()


def partial_spearman(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) if z.dtype != bool and len(np.unique(z)) > 2 else z.astype(float) for z in Z.T])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


res = {'layers': {}, 'cells': {}}
units_all, pushes_all = [], []
for L in LAYERS:
    U = pd.read_csv(f'{D}/L{L}_units.csv'); P = pd.read_csv(f'{D}/L{L}_pushes.csv')
    U['L'] = L; P['L'] = L
    n0 = len(P)
    drop = P[P['pre_ok'] == 0].groupby('group').size().to_dict()
    P = P[P['pre_ok'] != 0]
    units_all.append(U); pushes_all.append(P)
    cells = {}
    for g in GROUPS:
        d = P[P['group'] == g]
        if len(d) == 0: continue
        c = dict(n_units=int(d['uid'].nunique()), n_pts=len(d),
                 R2=r2yx(d['total_r'], d['meas_local_r']),
                 R2_ci=boot_units(d, 'total_r', 'meas_local_r', 1000),
                 R2_seed=[r2yx(d[d.seed == s]['total_r'], d[d.seed == s]['meas_local_r']) for s in sorted(d.seed.unique())],
                 R2_theory=float(1 - d['tr2'].sum() / d['r2'].sum()),
                 R2_rnd_push=r2yx(d['rnd_total_r'], d['rnd_meas_r']),
                 R2_fonly=r2yx(d['fonly_r'], d['meas_local_r']),
                 R2_graph_vs_e2e=r2yx(d['total_g'], d['e2e_real']),
                 R2_graphlocal=r2yx(d['total_g'], d['meas_local_g']),
                 spearman=float(spearmanr(d['total_r'], d['meas_local_r'])[0]),
                 med_relerr=float((np.abs(d.meas_local_r - d.total_r) / (np.abs(d.meas_local_r) + np.abs(d.total_r) + 1e-12)).median()),
                 R2_write_graph=r2yx(np.r_[-d['wr_pred'], d['wr_pred']], np.r_[d['wr_minus'], d['wr_plus']]),
                 R2_write_real=r2yx(np.r_[-d['wr_pred_r'], d['wr_pred_r']], np.r_[d['wr_minus'], d['wr_plus']]),
                 R2_write_rnd_real=r2yx(np.r_[-d['wr_rnd_pred_r'], d['wr_rnd_pred_r']], np.r_[d['wr_rnd_minus'], d['wr_rnd_plus']]),
                 mean_sd_meas=[float(d.meas_local_r.mean()), float(d.meas_local_r.std())],
                 g_cos_med=float(d['g_cos'].median()))
        cells[g] = c
    defined_all = r2yx(P['meas_local_r'], P['e2e_real'])
    defined_nooracle = r2yx(P[P.group != 'ORACLE']['meas_local_r'], P[P.group != 'ORACLE']['e2e_real'])
    N, R = cells['NEURON']['R2'], cells['REST']['R2']
    v = verdict(N, R)
    sv = [verdict(a, b) for a, b in zip(cells['NEURON']['R2_seed'], cells['REST']['R2_seed'])]
    if v in ('YES', 'NO') and sum(s == v for s in sv) < 2: v_final = 'MIXED'
    else: v_final = v
    if defined_all < 0.7: v_final = 'UNDEFINED'
    # class-label shuffle null for the gap NEURON - REST (permute unit labels within layer)
    d2 = P[P.group.isin(['NEURON', 'REST'])].copy()
    ulab = d2.groupby('uid')['group'].first()
    gaps = []
    obs_gap = N - R
    for _ in range(500):
        perm = pd.Series(rng.permutation(ulab.values), index=ulab.index)
        lab = d2['uid'].map(perm)
        a, b = d2[lab == 'NEURON'], d2[lab == 'REST']
        gaps.append(r2yx(a.total_r, a.meas_local_r) - r2yx(b.total_r, b.meas_local_r))
    gaps = np.array(gaps)
    res['layers'][L] = dict(n_pushes=n0, dropped_pre_ok=drop, defined_R2_all=defined_all, defined_R2_no_oracle=defined_nooracle,
                            pooled_verdict=v, seed_verdicts=sv, verdict=v_final, no_clause=no_clause(N, R) if v == 'NO' else [],
                            gap=obs_gap, shuffle_gap_q=[float(np.quantile(gaps, q)) for q in (.025, .5, .975)],
                            shuffle_p_two=float((np.abs(gaps) >= abs(obs_gap)).mean()))
    res['cells'][L] = cells
    print(L, json.dumps(res['layers'][L]), flush=True)
    for g, c in cells.items():
        print('   ', g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k not in ('R2_ci', 'R2_seed', 'mean_sd_meas')},
              'ci', [round(x, 3) for x in c['R2_ci']], 'seeds', [round(x, 3) for x in c['R2_seed']], flush=True)

vs = [res['layers'][L]['verdict'] for L in LAYERS]
res['status'] = 'CONFIRMED' if vs.count('YES') >= 3 else ('REJECTED' if vs.count('NO') >= 2 else 'UNCONFIRMED')
# Prediction A (no seed guard / gate in the prereg text? A = class clause on READ)
res['predA_layers_yes'] = sum(res['layers'][L]['pooled_verdict'] == 'YES' for L in LAYERS)
# Prediction B
U = pd.concat(units_all); P = pd.concat(pushes_all)
F = P[P.group.isin(['NEURON', 'GAP', 'REST'])]
re_read = relerr(F, 'total_r', 'meas_local_r')
Uf = U[U.group.isin(['NEURON', 'GAP', 'REST'])].drop_duplicates('uid').set_index('uid')
# uid may collide across layers: key by (L, uid)
F = F.assign(key=F.L.astype(str) + '_' + F.uid.astype(str))
Uf = U[U.group.isin(['NEURON', 'GAP', 'REST'])].assign(key=lambda x: x.L.astype(str) + '_' + x.uid.astype(str)).drop_duplicates('key').set_index('key')
def per_feat(pc, mc):
    a = np.abs(F[mc] - F[pc]) / (np.abs(F[mc]) + np.abs(F[pc]) + 1e-12)
    return a.groupby(F['key']).median()
F = F.assign(wrm_g=-F['wr_pred'], wrm_r=-F['wr_pred_r'])
Bres = {}
for name, (pc, mc) in {'READ': ('total_r', 'meas_local_r'), 'WRITE_graph': ('wrm_g', 'wr_minus'), 'WRITE_real': ('wrm_r', 'wr_minus')}.items():
    if pc not in F: continue
    e = per_feat(pc, mc)
    u = Uf.loc[e.index]
    Z = np.column_stack([np.log(u.fire_rate.values + 1e-9), u.mean_act.values, u.e_norm.values, u.d_norm.values] +
                        [(u.L.values == L).astype(float) for L in LAYERS[1:]])
    ps = partial_spearman(u.joint.values, e.values, Z)
    # shuffle null for the partial Spearman: permute joint within layer
    nulls = []
    for _ in range(300):
        jx = u.joint.values.copy()
        for L in LAYERS:
            m = u.L.values == L; jx[m] = rng.permutation(jx[m])
        nulls.append(partial_spearman(jx, e.values, Z))
    Bres[name] = dict(n_feat=len(e), partial_spearman=ps, raw_spearman=float(spearmanr(u.joint.values, e.values)[0]),
                      null_q=[float(np.quantile(nulls, q)) for q in (.025, .975)],
                      per_layer_partial={L: partial_spearman(u.joint.values[u.L.values == L], e.values[u.L.values == L], Z[u.L.values == L][:, :4]) for L in LAYERS})
res['predB'] = Bres
rd = Bres['READ']['partial_spearman']
wr = Bres.get('WRITE_graph', {}).get('partial_spearman', np.nan)
res['predB_verdict'] = 'HOLDS' if (rd <= -0.2 and abs(wr) < 0.1) else ('FAILS' if rd > -0.1 else 'UNSETTLED')
# Prediction C
cc = [(L, g) for L in LAYERS for g in ['NEURON', 'GAP', 'REST']]
diffs = {f'{L}_{g}': abs(res['cells'][L][g]['R2'] - res['cells'][L][g]['R2_theory']) for L, g in cc if g in res['cells'][L]}
nok = sum(v <= 0.15 for v in diffs.values())
res['predC'] = dict(diffs=diffs, n_within=nok, verdict='HOLDS' if nok >= 9 else ('FAILS' if nok <= 4 else 'UNSETTLED'))
print('STATUS', res['status'], 'layers', vs, 'B', res['predB_verdict'], json.dumps(Bres), 'C', res['predC'])
json.dump(res, open('vera_stats.json', 'w'), indent=1, default=float)
