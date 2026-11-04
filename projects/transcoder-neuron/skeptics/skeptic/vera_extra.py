"""Vera extra checks on Bruno's raw CSVs (own code): (1) raw (uncentred) cosine rule sensitivity,
(2) bootstrap CI of Prediction B's pooled partial Spearman, (3) dose-matched class R^2, (4) drop-top-k units,
(5) per-point vs per-feature median relative error."""
import json, numpy as np, pandas as pd
from scipy.stats import rankdata

D = '/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron/data/pushes'
rng = np.random.default_rng(99)
LAYERS = [8, 9, 10, 11]
cosdef = json.load(open('vera_cosdef.json'))


def r2yx(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def pspear(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) if len(np.unique(z)) > 2 else z for z in Z.T])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]; ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


out = {}
Us, Ps = [], []
for L in LAYERS:
    U = pd.read_csv(f'{D}/L{L}_units.csv'); P = pd.read_csv(f'{D}/L{L}_pushes.csv'); U['L'] = L; P['L'] = L
    Us.append(U); Ps.append(P)
    o = {}
    N = P[P.group == 'NEURON']; R = P[P.group == 'REST']
    # (3) dose-matched: strata = deciles of du_norm/xn_norm over NEURON+REST pushes; weight REST strata to NEURON's dose distribution
    both = pd.concat([N, R]); dr = both.du_norm / both.xn_norm
    edges = np.quantile(dr, np.linspace(0, 1, 11)); edges[-1] += 1
    bN = np.digitize(N.du_norm / N.xn_norm, edges); bR = np.digitize(R.du_norm / R.xn_norm, edges)
    o['dose_median_N_R'] = [float((N.du_norm / N.xn_norm).median()), float((R.du_norm / R.xn_norm).median())]
    # resample REST to NEURON's decile counts (and vice versa) -> R^2 with matched dose
    def matched(src, bsrc, tgt_b, reps=200):
        vals = []
        cnt = pd.Series(tgt_b).value_counts()
        for _ in range(reps):
            ii = []
            for b, c in cnt.items():
                pool = np.where(bsrc == b)[0]
                if len(pool): ii.append(rng.choice(pool, c, replace=True))
            ii = np.concatenate(ii)
            vals.append(r2yx(src.total_r.values[ii], src.meas_local_r.values[ii]))
        return [float(np.median(vals)), float(np.quantile(vals, .1)), float(np.quantile(vals, .9))]
    o['REST_R2_at_NEURON_dose'] = matched(R, bR, bN)
    o['NEURON_R2_at_REST_dose'] = matched(N, bN, bR)
    # common low-dose band: both below the NEURON 90th pct dose
    cut = np.quantile(N.du_norm / N.xn_norm, .9)
    nl, rl = N[N.du_norm / N.xn_norm <= cut], R[R.du_norm / R.xn_norm <= cut]
    o['low_dose_R2_N_R'] = [r2yx(nl.total_r, nl.meas_local_r), r2yx(rl.total_r, rl.meas_local_r), len(nl), len(rl)]
    # (4) drop top-k units by summed squared error
    for g, d in [('NEURON', N), ('REST', R)]:
        se = ((d.total_r - d.meas_local_r) ** 2).groupby(d.uid).sum().sort_values(ascending=False)
        o[f'{g}_drop_k'] = {k: r2yx(d[~d.uid.isin(se.index[:k])].total_r, d[~d.uid.isin(se.index[:k])].meas_local_r) for k in (0, 1, 2, 3, 5, 10)}
    # (5) relerr per point vs per feature
    for g, d in [('NEURON', N), ('REST', R)]:
        a = np.abs(d.meas_local_r - d.total_r) / (np.abs(d.meas_local_r) + np.abs(d.total_r) + 1e-12)
        o[f'{g}_medrel_point_feature'] = [float(a.median()), float(a.groupby(d.uid).median().median())]
    # (1) raw-cosine rule: drop sampled NEURON units that are not NEURON under the uncentred rule (count from vera_cosdef)
    o['raw_rule_neuron_rule_ok'] = cosdef[str(L)]['raw']['neuron_rule_ok']
    out[L] = o
    print(L, json.dumps(o), flush=True)

# (2) bootstrap the pooled partial Spearman of Prediction B (features resampled with replacement)
U = pd.concat(Us); P = pd.concat(Ps)
F = P[P.group.isin(['NEURON', 'GAP', 'REST'])].copy(); F['key'] = F.L.astype(str) + '_' + F.uid.astype(str)
a = np.abs(F.meas_local_r - F.total_r) / (np.abs(F.meas_local_r) + np.abs(F.total_r) + 1e-12)
e = a.groupby(F.key).median()
Uf = U[U.group.isin(['NEURON', 'GAP', 'REST'])].assign(key=lambda x: x.L.astype(str) + '_' + x.uid.astype(str)).drop_duplicates('key').set_index('key').loc[e.index]
Z = np.column_stack([np.log(Uf.fire_rate.values + 1e-9), Uf.mean_act.values, Uf.e_norm.values, Uf.d_norm.values] + [(Uf.L.values == L).astype(float) for L in LAYERS[1:]])
x, y = Uf.joint.values, e.values
obs = pspear(x, y, Z)
bs = []
for _ in range(1000):
    ii = rng.integers(0, len(x), len(x)); bs.append(pspear(x[ii], y[ii], Z[ii]))
bs = np.array(bs)
out['predB_boot'] = dict(obs=obs, ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         P_below_minus0p1=float((bs <= -0.1).mean()), P_below_minus0p2=float((bs <= -0.2).mean()))
# law without layer dummies, and Spearman only within the gap zone (continuum test)
gz = Uf.group.values == 'GAP'
out['predB_gapzone_only_partial'] = pspear(x[gz], y[gz], Z[gz])
print(json.dumps(out['predB_boot']), 'gap-zone-only partial', out['predB_gapzone_only_partial'])
json.dump(out, open('vera_extra.json', 'w'), indent=1, default=float)
