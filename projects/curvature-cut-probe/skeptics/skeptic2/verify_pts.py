"""Hugo (skeptic2): re-check saved found moves independently of the search code path.
For each saved point z (float32 on disk): (1) on the ellipsoid? ||z/gamma|| vs ||z0/gamma||;
(2) score shift with the probe formula; (3) KL(p0||softmax(W z)) in float64 from the safetensors W
(loaded fresh in 16k-row chunks; no shared helper). Prints the cheapest point that reaches each false
statement's distance D to the clean line.
usage: verify_pts.py out/ellip_pts_<arm>_off0_n4.npz ARM
"""
import json
import os
import sys

import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
import glob
SNAP = glob.glob('/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors')[0]

f, arm = sys.argv[1], sys.argv[2]
o = np.load(f)
idx, Zp = o['idx'], o['z'].astype(np.float64)  # (nlam, B, d)
split = json.load(open(os.path.join(DATA, 'ref/split.json')))
zall = np.load(os.path.join(DATA, 'clean/z_post.npy')).astype(np.float64)
pz = np.load(os.path.join(DATA, 'fits/probes.npz'))
w, mu, sd = [pz[f'{arm}__{k}'].astype(np.float64) for k in ('w', 'mu', 'sd')]
b = float(pz[f'{arm}__b'])
score = lambda z: ((z - mu) / sd) @ w + b
tr = np.array([x['true'] for x in split]); sp = np.array([x['split'] for x in split])
line = np.sort(score(zall)[(sp == 'refit') & (tr == 1)])[::-1][1]
with safe_open(SNAP, 'pt') as s:
    gam = s.get_tensor('model.norm.weight').double().numpy()
    Wt = s.get_slice('model.embed_tokens.weight')
    V = Wt.get_shape()[0]
    Z0 = zall[idx]
    allz = np.concatenate([Z0[None], Zp], 0).reshape(-1, Z0.shape[1])  # (1+nlam)*B
    L = np.empty((allz.shape[0], V))
    for a in range(0, V, 16384):
        Wc = Wt[a:a + 16384].double().numpy()
        L[:, a:a + len(Wc)] = allz @ Wc.T
L -= L.max(1, keepdims=True)
LQ = L - np.log(np.exp(L).sum(1, keepdims=True))
B = len(idx)
LQ = LQ.reshape(1 + Zp.shape[0], B, -1)
lp0 = LQ[0]
for i in range(B):
    r0 = np.linalg.norm(Z0[i] / gam)
    rs = [np.linalg.norm(Zp[j, i] / gam) for j in range(Zp.shape[0])]
    kl = [(np.exp(lp0[i]) * (lp0[i] - LQ[1 + j, i])).sum() for j in range(Zp.shape[0])]
    sh = [score(Z0[i]) - score(Zp[j, i]) for j in range(Zp.shape[0])]
    t = split[idx[i]]['true']
    D = score(Z0[i]) - line if t == 0 else None
    msg = f'idx {idx[i]} true {t} |y0| {r0:.4f} max| |y|-|y0| | {max(abs(x - r0) for x in rs):.2e}'
    if t == 0 and D > 0:
        ok = [(k, s_) for k, s_ in zip(kl, sh) if s_ >= D]
        msg += f' D {D:.3f} cheapest KL reaching D: {min(ok)[0]:.3e}' if ok else f' D {D:.3f} not reached'
    print(msg)
    print('   shift', np.round(sh, 3)[:8], '\n   kl   ', np.array(kl)[:8])
