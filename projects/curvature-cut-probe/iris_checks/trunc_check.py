"""Iris: does truncating the output Fisher to the top-K tokens change the flat share of the probe?
Post-norm country-token states (readout 29, clean, saved by theory-1022), tok29 probe from panel.npz.
F_full = W^T (diag p - p p^T) W over all 151,936 tokens (float64, streamed); F_K = same over the top-K tokens of p
(p NOT renormalised, as a truncated sum). Reports tail mass, ||F_full - F_K||_2, and the share of u = w/sd in
eigen-directions below absolute cuts, for both. 6 eval FALSE statements (seed 0). CPU only.
Run: venv1024/bin/python iris_checks/trunc_check.py"""
import json, os, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
IN = HERE / 'inputs'
SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
V, D, CH = 151936, 1536, 8192
f = safe_open(str(SNAP), 'pt')
E = f.get_slice('model.embed_tokens.weight')


def chunks():
    for a in range(0, V, CH):
        yield a, E[a:min(V, a + CH)].double().numpy()


SPLIT = json.load(open(IN / 'split.json'))
y = np.array([1 - r['true'] for r in SPLIT])
split = np.array([r['split'] for r in SPLIT])
evf = np.where((split == 'eval') & (y == 1))[0]
Z = np.load(IN / 'clean_country_R29.npy').astype(np.float64)
z = np.load(IN / 'panel.npz', allow_pickle=True)
u = z['tok29__w'].astype(np.float64) / z['tok29__sd'].astype(np.float64)
sel = np.random.default_rng(0).choice(evf, 6, replace=False)
X = Z[sel]
L = np.concatenate([X @ Ec.T for _, Ec in chunks()], 1)
L -= L.max(1, keepdims=True)
P = np.exp(L)
P /= P.sum(1, keepdims=True)
CUTS = [1e-8, 1e-6, 1e-4]
rows = []
t0 = time.time()
for k in range(len(sel)):
    p = P[k]
    order = np.argsort(p)[::-1]
    res = {'idx': int(sel[k]), 'top_p': float(p[order[0]])}
    for K in (8192, 32768, V):
        keep = np.zeros(V, bool)
        keep[order[:K]] = True
        M = np.zeros((D, D))
        m = np.zeros(D)
        for a, Ec in chunks():
            pc = p[a:a + len(Ec)] * keep[a:a + len(Ec)]
            M += (Ec.T * pc) @ Ec
            m += Ec.T @ pc
        M -= np.outer(m, m)
        lam, U = np.linalg.eigh(M)
        sh = (U.T @ u) ** 2 / (u @ u)
        r = {'tail_mass': float(p[~keep].sum()), 'lam_max': float(lam[-1])}
        for c in CUTS:
            r[f'share_u_lt_{c:g}'] = float(sh[lam < c].sum())
            r[f'n_lt_{c:g}'] = int((lam < c).sum())
        r['uFu_over_uu'] = float(u @ M @ u / (u @ u))
        res[f'K{K}'] = r
        if K == V:
            Mfull = M
        else:
            res[f'K{K}']['_M'] = M
    for K in (8192, 32768):
        res[f'K{K}']['opnorm_diff'] = float(np.linalg.norm(Mfull - res[f'K{K}'].pop('_M'), 2))
    rows.append(res)
    print(json.dumps(res), flush=True)
out = {'rows': rows, 'seconds': round(time.time() - t0, 1),
       'emb_rownorm_sq_quantiles': None}
nrm = np.concatenate([(Ec ** 2).sum(1) for _, Ec in chunks()])
out['emb_rownorm_sq_quantiles'] = dict(zip(['p50', 'p99', 'max'], np.quantile(nrm, [.5, .99, 1]).tolist()))
json.dump(out, open(HERE / 'trunc_check.json', 'w'), indent=1)
print(out['emb_rownorm_sq_quantiles'], out['seconds'])
