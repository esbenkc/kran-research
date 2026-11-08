"""How much of the probe's per-token footprint is SYSTEMATIC (same sign/size for a token id across prompts) vs
hash-like? Parent rec_s2/rec_s3 answer-slot top-20 logits (served none vs leaky). Split each prompt's centred
logit change d_i(v) into mean_v (over prompts, per token id) + residual. TV from the systematic part only =
what a reader that knows token identities (not the rounding) could at best exploit, per token."""
import numpy as np, os, json
from collections import defaultdict
P = os.path.expanduser('~/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/data/status')
rows = []
for run in ['rec_s2', 'rec_s3']:
    a = np.load(f'{P}/{run}_raw/none_plain.npz'); b = np.load(f'{P}/{run}_raw/ref_plain.npz')
    for i in range(200):
        ia, ib = list(a['top20_id'][i]), list(b['top20_id'][i])
        ids = [t for t in ia if t in ib]
        la = np.array([a['top20_logit'][i][ia.index(t)] for t in ids], float)
        lb = np.array([b['top20_logit'][i][ib.index(t)] for t in ids], float)
        p = np.exp(la - float(a['lse'][i])); d = lb - la
        d = d - (p * d).sum() / p.sum()          # centre under p (a uniform shift changes nothing)
        rows.append((run, ids, p, d))
acc = defaultdict(list)
for _, ids, p, d in rows:
    for t, x in zip(ids, d): acc[t].append(x)
mean = {t: np.mean(v) for t, v in acc.items() if len(v) >= 20}
def tv(p, d):
    q = p * np.exp(d); q = q / q.sum() * p.sum()
    return 0.5 * np.abs(p - q).sum()
full, sysm, resid, n_cov = [], [], [], []
for _, ids, p, d in rows:
    m = np.array([mean.get(t, 0.0) for t in ids])
    full.append(tv(p, d)); sysm.append(tv(p, m)); resid.append(tv(p, d - m))
    n_cov.append(sum(p[[t in mean for t in ids]]) / p.sum())
var_tot = np.mean([np.sum(p * d * d) / p.sum() for _, _, p, d in rows])
var_sys = np.mean([np.sum(p * np.array([mean.get(t, 0.0) for t in ids]) ** 2) / p.sum() for _, ids, p, d in rows])
out = dict(tv_full=float(np.mean(full)), tv_systematic=float(np.mean(sysm)), tv_residual=float(np.mean(resid)),
           p_weighted_var_total=float(var_tot), p_weighted_var_systematic=float(var_sys),
           systematic_var_share=float(var_sys / var_tot), mass_on_tokens_with_mean=float(np.mean(n_cov)),
           n_tokens_with_mean=len(mean), n_rows=len(rows))
print({k: round(v, 5) if isinstance(v, float) else v for k, v in out.items()})
for t in sorted(mean, key=lambda t: -len(acc[t]))[:8]:
    v = np.array(acc[t]); print('token', t, 'n', len(v), 'mean', round(v.mean(), 4), 'sd', round(v.std(), 4))
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'systematic_tv.json'), 'w'), indent=1)
