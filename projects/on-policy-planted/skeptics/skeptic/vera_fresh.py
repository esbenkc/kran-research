"""Vera: fresh trace read at 1,200 (800 thoughts sampled from adapter_1200, probe-on likelihood ratio recomputed under
that checkpoint) from data/pool/fresh1200/cols.npz + data/reads/fresh1200/c_1200.npy. Same estimators as vera_base.
Paired D_R(fresh) - D_R(stored) on the matching stored rows (same prompt, same draw index), prompt-cluster bootstrap."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_base as vb

P = os.path.expanduser('~/work/oracle-1113-on-policy-planted/kran-research/projects/on-policy-planted')


def load(pool):
    vb.P = P
    orig = f'{P}/data/pool/test/cols.npz'
    d = np.load(f'{P}/data/pool/{pool}/cols.npz')
    v = d['variant']; base = d['base_row']; toks = d['toks']
    pl = np.where(v == 0)[0]; tw_of = {int(base[j]): int(j) for j in np.where(v == 1)[0]}
    key = lambda i: int(base[i]) if int(base[i]) in tw_of else int(i)
    pl = np.array([i for i in pl if key(i) in tw_of]); tw = np.array([tw_of[key(i)] for i in pl])
    return d, dict(pl=pl, tw=tw, prompt=d['prompt_row'][pl], logw=d['logw'][pl].astype(np.float64),
                   x1=d['x1_orig'][pl].astype(bool), draw=d['draw'][pl] if 'draw' in d.files else None,
                   checks=dict(n_plain=int(len(pl)), twin_rest_equal=bool((toks[pl, 1:] == toks[tw, 1:]).all()),
                               twin_token0_is_X=bool((toks[tw, 0] == 1205).all()), n_x1=int(d['x1_orig'][pl].sum())))


df, F = load('fresh1200')
bt = vb.Boot(F['prompt'])
res = dict(checks=F['checks'])
for tag in ('0', '1200'):
    c = np.load(f'{P}/data/reads/fresh1200/c_{tag}.npy').astype(np.float64)
    st = vb.stats(F, c, bt)
    res[f'adapter_{tag}'] = {k: vb.pc(st[k]) for k in ('cheat_off', 's_P', 'y_R', 'D_R', 's_R')}
# paired D_R vs stored rows with the same (prompt, draw)
ds, S = load('test')
sd = {(int(p), int(r)): i for i, (p, r) in enumerate(zip(S['prompt'], ds['draw'][S['pl']] if 'draw' in ds.files else
                                                       np.zeros(len(S['pl']))))}
fd = df['draw'][F['pl']]
idx_s = [sd.get((int(p), int(r))) for p, r in zip(F['prompt'], fd)]
ok = np.array([i is not None for i in idx_s])
res['paired_rows'] = int(ok.sum())
if ok.all():
    ws = np.exp(S['logw'][np.array(idx_s)]); wf = np.exp(F['logw'])
    diff = np.maximum(0, 1 - wf) - np.maximum(0, 1 - ws)
    n = bt.sums(np.ones(len(diff)))
    res['D_R_fresh_minus_stored'] = vb.pc(bt.sums(diff) / n)
    res['D_R_stored_same_rows'] = round(float(np.maximum(0, 1 - ws).mean()), 4)
    res['prompt_aligned'] = bool((S['prompt'][np.array(idx_s)] == F['prompt']).all())
    # how far the fresh thoughts moved from their stored twin (same prompt, same draw, same noise seed)
    tf, ts = df['toks'][F['pl']], ds['toks'][S['pl']][np.array(idx_s)]
    eq = tf == ts
    res['fresh_identical_share'] = round(float(eq.all(1).mean()), 4)
    res['fresh_token_equal_share'] = round(float(eq.mean()), 4)
    res['fresh_median_first_diff'] = int(np.median(np.where(eq.all(1), 64, np.argmin(eq, 1))))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_fresh.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
