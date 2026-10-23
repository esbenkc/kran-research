"""Idealized attacker, closed form (owner Iris). Same 16 twin pairs and vocabulary cut as theory/flow_sim.py, but the
output penalty is replaced by its second-order form KL ~ 1/2 x^T F x (F = output Fisher at the clean state, exact
RMSNorm Jacobian, so F h0 = 0 exactly along the free direction). Per pair, with x = (x_false, x_true) the moves:
    L(x) = gap(x)^2 + lam/2 (x_f^T F_f x_f + x_t^T F_t x_t),  gap(x) = g0 + u^T x,  u = (w, -w)
    gradient flow dx/dt = -(A x + 2 g0 u),  A = 2 u u^T + lam blockdiag(F_f, F_t)
    x(t) = -sum_k (1 - exp(-a_k t)) (v_k^T 2 g0 u / a_k) v_k          (eigenpairs a_k, v_k of A; a_k > 0)
As t -> infinity the flow reaches the nearest zero-loss point, which for lam > 0 is a PURE length change (R = 1),
because the length is the only exactly free direction. The question is the time scale. We report R (frozen metric,
exact nonlinear restore) at the flow time where the mean gap has fallen 50/90/95/99% (the LoRA's precondition is 50%),
for lam from 1e2 to 1e10. Checked against the exact-KL gradient descent of flow_sim.py at the lam values both ran.
CPU float64, numpy. Run (cwd = this project): ../../../venv1023/bin/python theory/flow_quadratic.py"""
import json, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
EPS, V, D, CH, N_PAIRS, P_MIN = 1e-6, 151936, 1536, 8192, 16, 1e-6
LAMS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10]
DROPS = [0.5, 0.9, 0.95, 0.99]

SPLIT = json.load(open(ROOT / 'data' / 'ref' / 'split.json'))
y = np.array([1 - r['true'] for r in SPLIT]); sp_ = np.array([r['split'] for r in SPLIT])
city = np.array([r['city'] for r in SPLIT]); src = np.array([r['src'] for r in SPLIT])
pairs = []
for i in np.where((sp_ == 'eval') & (y == 1))[0]:
    j = np.where((sp_ == 'eval') & (y == 0) & (city == city[i]) & (src == src[i]))[0]
    pairs.append((i, int(j[0])))
pairs = np.array(pairs)
sel = pairs[np.random.default_rng(0).choice(len(pairs), N_PAIRS, replace=False)]   # same 16 pairs as flow_sim.py
Hall = np.load(ROOT / 'theory' / 'inputs' / 'clean_country_R28.npy').astype(np.float64)
z = np.load(ROOT / 'data' / 'fits' / 'panel.npz', allow_pickle=True)
w = z['tok28__w'].astype(np.float64) / z['tok28__sd']

import torch  # only to read the bf16 tensors  # noqa: E402
f = safe_open(str(SNAP), 'pt')
g = f.get_tensor('model.norm.weight').double().numpy()
E = f.get_slice('model.embed_tokens.weight')


def nrm(X):
    return g * X / np.sqrt((X * X).mean(-1, keepdims=True) + EPS)


idx = np.r_[sel[:, 0], sel[:, 1]]
X0 = Hall[idx]
Z = np.concatenate([nrm(X0) @ E[a:min(V, a + CH)].double().numpy().T for a in range(0, V, CH)], 1)
Z -= Z.max(1, keepdims=True)
P = np.exp(Z); P /= P.sum(1, keepdims=True)
keep = (P > P_MIN).any(0)
Wk = np.concatenate([E[a:min(V, a + CH)].double().numpy()[keep[a:min(V, a + CH)]] for a in range(0, V, CH)], 0)
del Z


def fisher(h):
    zz = nrm(h[None]) @ Wk.T
    p = np.exp(zz[0] - zz.max()); p /= p.sum()
    r = np.sqrt((h * h).mean() + EPS)
    J = g[:, None] * (np.eye(D) / r - np.outer(h, h) / (D * r ** 3))   # dN/dh
    Wp = Wk.T @ p
    M = (Wk.T * p) @ Wk - np.outer(Wp, Wp)
    return J.T @ M @ J


t0 = time.time()
hf, ht = X0[:N_PAIRS], X0[N_PAIRS:]
g0 = hf @ w - ht @ w
u = np.r_[w, -w]
T = np.logspace(-4, 14, 721)
NL = len(LAMS)
GAP = np.zeros((NL, N_PAIRS, len(T))); GR = np.zeros_like(GAP); KL = np.zeros_like(GAP); NRF = np.zeros_like(GAP)
rad = []
for k in range(N_PAIRS):                       # pairs outer: only two Fisher matrices in memory at a time
    Ff, Ft = fisher(hf[k]), fisher(ht[k])
    rad.append(max(abs(hf[k] @ Ff @ hf[k]) / (hf[k] @ hf[k]), abs(ht[k] @ Ft @ ht[k]) / (ht[k] @ ht[k])))
    for li, lam in enumerate(LAMS):
        A = 2 * np.outer(u, u)
        A[:D, :D] += lam * Ff; A[D:, D:] += lam * Ft
        a, Vv = np.linalg.eigh(A)
        c = Vv.T @ (-2 * g0[k] * u)
        ok = a > 1e-12 * a.max()
        Em = -np.expm1(-np.outer(a[ok], T))                 # 1 - exp(-a t)
        Xt = Vv[:, ok] @ ((c[ok] / a[ok])[:, None] * Em)     # (2D, len(T))
        del Vv, Em
        xf, xt = Xt[:D], Xt[D:]
        Hf, Ht = hf[k][:, None] + xf, ht[k][:, None] + xt
        GAP[li, k] = w @ Hf - w @ Ht
        rf = np.linalg.norm(hf[k]) / np.linalg.norm(Hf, axis=0); rt = np.linalg.norm(ht[k]) / np.linalg.norm(Ht, axis=0)
        GR[li, k] = rf * (w @ Hf) - rt * (w @ Ht)
        KL[li, k] = 0.5 * (np.einsum('it,ij,jt->t', xf, Ff, xf) + np.einsum('it,ij,jt->t', xt, Ft, xt)) / 2
        NRF[li, k] = 1 / rf
    print(f'pair {k} done {time.time() - t0:.0f}s', flush=True)
out = {'n_pairs': N_PAIRS, 'kept_tokens': int(keep.sum()), 'g0_mean': float(g0.mean()),
       'radial_rayleigh_max': float(max(rad)), 'kl_note': 'kl_mean = mean over the 32 states of 1/2 x^T F x',
       'lams': {}}
for li, lam in enumerate(LAMS):
    gc, ga, gr = g0.mean(), GAP[li].mean(0), GR[li].mean(0)
    drop = (gc - ga) / gc
    R = (gr - ga) / np.where(gc - ga != 0, gc - ga, np.nan)
    row = {}
    for dr in DROPS:
        j = int(np.argmax(drop >= dr)) if (drop >= dr).any() else None
        row[f'drop{dr:g}'] = None if j is None else {'t': float(T[j]), 'R': float(R[j]), 'kl_mean': float(KL[li, :, j].mean()),
                                                      'norm_ratio_false_median': float(np.median(NRF[li, :, j]))}
    row['R_t_end'] = float(R[-1]); row['drop_t_end'] = float(drop[-1])
    row['curve'] = {'t': T[::10].tolist(), 'R': R[::10].tolist(), 'drop': drop[::10].tolist()}
    out['lams'][f'{lam:g}'] = row
    print(f'lam {lam:g} ' + json.dumps({k: v for k, v in row.items() if k != 'curve'}), flush=True)
OUTP = ROOT / 'data' / 'results' / 'flow_quadratic.json'
json.dump(out, open(OUTP, 'w'), indent=1)
print('wrote', OUTP)
