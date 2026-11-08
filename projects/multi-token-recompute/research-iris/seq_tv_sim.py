"""Anchor (not data): if each of k prefix tokens had the answer-slot sensitivity of the parent's clean-rerun
adapters (rec_s2/rec_s3, served none vs leaky top-20 logits), what are sequence TV, union bound and
Gumbel-coupled flip rate f? Tokens drawn independently from the 200 per-prompt (p, q) pairs."""
import numpy as np, os, json
P = os.path.expanduser('~/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/data/status')
rng = np.random.default_rng(0)
pairs = []
for run in ['rec_s2', 'rec_s3']:
    a = np.load(f'{P}/{run}_raw/none_plain.npz'); b = np.load(f'{P}/{run}_raw/ref_plain.npz')
    for i in range(200):
        ia, ib = list(a['top20_id'][i]), list(b['top20_id'][i])
        ids = [t for t in ia if t in ib]
        p = np.exp([a['top20_logit'][i][ia.index(t)] - a['lse'][i] for t in ids]).astype(float)
        q = np.exp([b['top20_logit'][i][ib.index(t)] - b['lse'][i] for t in ids]).astype(float)
        p = np.append(p, max(1 - p.sum(), 1e-12)); q = np.append(q, max(1 - q.sum(), 1e-12))
        pairs.append((p / p.sum(), q / q.sum()))
tv_tok = np.array([0.5 * np.abs(p - q).sum() for p, q in pairs])
kl_tok = np.array([(p * np.log(p / q)).sum() for p, q in pairs])
res = {'tv_tok_mean': tv_tok.mean(), 'kl_tok_mean': kl_tok.mean()}
N = 20000
for k in [1, 4, 16, 64, 256]:
    L = np.zeros(N); flip = np.zeros(N, bool)
    for t in range(k):
        idx = rng.integers(len(pairs), size=N)
        for j in range(N):
            p, q = pairs[idx[j]]
            g = rng.gumbel(size=len(p))
            x = np.argmax(np.log(p) + g); y = np.argmax(np.log(q) + g)
            flip[j] |= (x != y) if not flip[j] else True
            L[j] += np.log(q[x] / p[x])
    tv_seq = np.mean(np.clip(1 - np.exp(L), 0, None))
    res[f'k{k}'] = dict(tv_seq=tv_seq, union=min(1, k * tv_tok.mean()), pinsker=min(1, np.sqrt(k * kl_tok.mean() / 2)),
                        f_gumbel=flip.mean())
    print(k, {kk: round(v, 4) for kk, v in res[f'k{k}'].items()})
print('per-token TV', round(tv_tok.mean(), 4), 'KL', round(kl_tok.mean(), 6))
json.dump(res, open(os.path.join(os.path.dirname(__file__), 'seq_tv_sim.json'), 'w'), indent=1, default=float)
