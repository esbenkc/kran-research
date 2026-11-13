"""Vera: row-level checks from batches/*.npz + steps.jsonl (dec in batch order: 32 probe-off rows, then 32 probe-on;
dec 0 = read/cheat, verified: steps 'cheat' = 1 - mean(dec)).
1) row alignment: corr(dec, pilot decision logit diff d2_ref[:,0]-d2_ref[:,1]) as saved vs rows shuffled.
2) reward signal: shaped advantage rebuilt from the frozen recipe (read 1, solve 0.6; probed: r - lam_g z(M at the
   sampled decision), lam_g = 0.42 min(1, sd_g(r)/0.2); advantage = minus group mean, no std division).
3) training-time switch: cheat(no We) - cheat(We) on probed fresh rows, prompt-clustered CI.
4) cached d010_s0 training switch per block (probed no-X minus X), from the parent's curve.jsonl."""
import json, os
import numpy as np

P = os.path.expanduser('~/work/oracle-1113-on-policy-planted/kran-research/projects/on-policy-planted')
PP = os.path.expanduser('~/work/oracle-1113-decision-only/kran-research/projects/decision-only-control')
B = f'{P}/data/runs/live_k4/batches'
st = [json.loads(l) for l in open(f'{P}/data/runs/live_k4/steps.jsonl')]
_c = {}
def get(kind, step):
    if (kind, step) not in _c:
        _c[(kind, step)] = np.load(f'{B}/{kind}_{step:05d}.npz')
    return _c[(kind, step)]

out = {}
assert all(abs(s['cheat'] - (1 - np.mean(s['dec']))) < 1e-6 for s in st)
rng = np.random.default_rng(0); xs, ys, xsh = [], [], []
for s in st:
    dd = np.concatenate([get('off', s['born_off'])['d2_ref'], get('on', s['born_on'])['d2_ref']])
    d = dd[:, 0] - dd[:, 1]; xs.append(d); ys.append(np.array(s['dec'])); xsh.append(d[rng.permutation(64)])
X, Y, XS = map(np.concatenate, (xs, ys, xsh))
out['align_corr'] = round(float(np.corrcoef(X, Y)[0, 1]), 4); out['align_corr_shuffled'] = round(float(np.corrcoef(XS, Y)[0, 1]), 4)

rows = []
for s in st:
    for kind, sl in (('on', slice(32, 64)), ('off', slice(0, 32))):
        b = get(kind, s[f'born_{kind}']); dec = np.array(s['dec'][sl]); read = dec == 0
        x = b['toks'][:, 0] == 1205; r = np.where(read, 1.0, 0.6); A = np.zeros(32); pk = np.repeat(b['prompts'], 8)
        for g in range(4):
            q = slice(8 * g, 8 * g + 8); rr = r[q]
            if kind == 'on':
                m = np.where(read, b['M_read'], b['M_solve']).astype(float)[q]; sd = m.std()
                z = (m - m.mean()) / sd if sd > 0 else np.zeros(8)
                rp = rr - 0.42 * min(1.0, rr.std() / 0.2) * z
            else:
                rp = rr
            A[q] = rp - rp.mean()
        for i in range(32):
            rows.append((kind == 'on', pk[i], read[i], x[i], A[i]))
R = np.array(rows, dtype=float)
for on in (1, 0):
    for xv in (1, 0):
        m = (R[:, 0] == on) & (R[:, 3] == xv)
        out[f'adv_{"probed" if on else "unprobed"}_{"X" if xv else "noX"}'] = dict(
            A_read=round(float(R[m & (R[:, 2] == 1), 4].mean()), 4), A_solve=round(float(R[m & (R[:, 2] == 0), 4].mean()), 4),
            n=int(m.sum()))
on = R[R[:, 0] == 1]; xm = on[:, 3] == 1
out['train_switch_noX_minus_X'] = round(float(on[~xm, 2].mean() - on[xm, 2].mean()), 4)
u, inv = np.unique(on[:, 1], return_inverse=True); idx = [np.where(inv == g)[0] for g in range(len(u))]
rng = np.random.default_rng(0); v = []
for _ in range(1000):
    ii = np.concatenate([idx[g] for g in rng.integers(0, len(u), len(u))]); rr = on[ii]; m = rr[:, 3] == 1
    v.append(rr[~m, 2].mean() - rr[m, 2].mean())
out['train_switch_ci'] = [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
out['train_prompts'] = int(len(u)); out['X_row_uses_probed'] = int(xm.sum())

cc = [json.loads(l) for l in open(f'{PP}/data/runs/d010_s0/curve.jsonl')]
prev = 0; blocks = {}
for c in cc:
    N = 32 * (c['step'] - prev); nx = c['n_on_x']; prev = c['step']
    notx = (c['cheat_on'] * N - c['train_cheat_on_x'] * nx) / (N - nx)
    blocks[c['step']] = round(float(notx - c['train_cheat_on_x']), 3)
out['cached_train_switch_by_block'] = blocks
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_extra.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
