"""critic2 (Claude restart) independent rebuild of the headline cells from saved raw data.
Own code; reads only data/ref/split.json, data/scores/*.npz, data/states/*/{kl,lp,off}.npy.
Rule B: each member tok10..tok29 re-standardised on the state's 200 'refit' true statements (mean, sd ddof0),
max over members, line = 2nd-highest of those 200, flag iff score > line. Heads: sklearn with PREREG s5 settings."""
import json, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
S = json.load(open('data/ref/split.json'))
sp = np.array([d['split'] for d in S]); y = np.array([d['true'] for d in S])
LINE = (sp == 'refit') & (y == 1); EVF = (sp == 'eval') & (y == 0); EVT = (sp == 'eval') & (y == 1)
TR = (sp == 'probe') | (sp == 'stop')
assert LINE.sum() == 200 and EVF.sum() == 596 and EVT.sum() == 596 and TR.sum() == 800
MEM = [f'tok{i}' for i in range(10, 30)]
def line2(v): return np.sort(v)[-2]
def run(st):
    z = np.load(f'data/scores/{st}.npz'); X = np.stack([z[m] for m in MEM], 1)
    mu, sd = X[LINE].mean(0), X[LINE].std(0); B = ((X - mu) / sd).max(1); t = line2(B[LINE])
    o = {'ruleB_catch_n': int((B[EVF] > t).sum()), 'ruleB_fa_n': int((B[EVT] > t).sum())}
    sc = StandardScaler().fit(X[TR]); Xs = sc.transform(X)
    for name, m in [('logit', LogisticRegression(C=1.0, max_iter=5000)), ('qda', QDA(reg_param=0.1))]:
        s = m.fit(Xs[TR], y[TR]).predict_proba(Xs)[:, 0]  # P(false)
        tt = line2(s[LINE]); o[f'{name}_catch_n'] = int((s[EVF] > tt).sum()); o[f'{name}_fa_n'] = int((s[EVT] > tt).sum())
    rng = np.random.default_rng(1); ysh = rng.permutation(y[TR])
    s = LogisticRegression(C=1.0, max_iter=5000).fit(Xs[TR], ysh).predict_proba(Xs)[:, 0]; tt = line2(s[LINE])
    o['logit_shuffled_catch_n'] = int((s[EVF] > tt).sum())
    d = f'data/states/{st}'
    try:
        kl, lp, off = (np.load(f'{d}/{f}.npy') for f in ('kl', 'lp', 'off'))
        ev = np.where(sp == 'eval')[0]
        o['kl_eval_tokmean'] = float(np.concatenate([kl[off[i]:off[i+1]] for i in ev]).mean())
        o['kl_all_tokmean'] = float(kl.mean())
        # twin pairs: eval 'cities' true vs false of same city; compare summed statement log-prob
        tot = np.array([lp[off[i]:off[i+1]].sum() for i in range(len(S))])
        idx = {}
        for i, dd in enumerate(S):
            if dd['split'] == 'eval' and dd['src'] == 'cities': idx.setdefault(dd['city'], {})[dd['true']] = i
        pr = [(v[1], v[0]) for v in idx.values() if 0 in v and 1 in v]
        o['pairs_right'] = int(sum(tot[a] > tot[b] for a, b in pr)); o['n_pairs'] = len(pr)
        o['kl_meta'] = json.load(open(f'{d}/meta.json')).get('kl_eval_mean_every_position')
    except FileNotFoundError: pass
    return o
R = {st: run(st) for st in ['clean', 'recal__lam10000__seed0', 'recal__lam10000__seed1', 'recal__lam10000__seed2', 'recalSG__lam10000__seed0']}
for st, o in R.items():
    print(st, ' '.join(f'{k}={v/596*100:.1f}%' if k.endswith('_n') else f'{k}={v}' for k, v in o.items()))
IL = [R[f'recal__lam10000__seed{i}'] for i in range(3)]
pool = {k: sum(o[k] for o in IL) / 1788 * 100 for k in ['ruleB_catch_n', 'ruleB_fa_n', 'logit_catch_n', 'logit_fa_n', 'qda_catch_n', 'qda_fa_n', 'logit_shuffled_catch_n']}
pool['kl_meta_mean'] = float(np.mean([o['kl_meta'] for o in IL])); pool['kl_eval_tokmean_mean'] = float(np.mean([o['kl_eval_tokmean'] for o in IL]))
print('POOLED in-loop 3 seeds:', json.dumps({k: round(v, 5) for k, v in pool.items()}))
json.dump({'per_state': R, 'pooled_inloop': pool}, open('skeptics/critic2/c2_rebuild.json', 'w'), indent=1)
