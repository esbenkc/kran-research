"""Vera attack on '<=200 numbers cannot reach the floor': a family nobody tried, low-rank factors.
W_hh = U V^T (rank r), W_oh = B C^T (rank k; the softmax shift 1 a^T is argmax-free and dropped).
W_hi is trained, then is exactly removable by positive per-neuron scaling to +-1 (keeps both ranks), so it
costs 0 numbers. Free numbers (GL gauge removed): (32 r - r^2) + (26 k - k^2). Warm start = truncated SVD in the rms gauge.
KL to the model's logits, fresh N(0,1) batches (torch seed 920+cfg), eval on dev seed 921 (2e5). Fitted network.
Run: cd swarm-algzoo-432 && .venv/bin/python skeptic/vera_lowrank.py -> skeptic/vera_lowrank.json
"""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_core import load, pred, sample, HERE

torch.set_num_threads(4)
Wi, Wh, Wo = load()
# rms gauge: scale neuron i by s_i so its typical activation is 1 (exact for ReLU, no biases)
Xg = sample(919, 20000)
H = np.zeros((len(Xg), 16)); acts = []
for t in range(10):
    H = np.maximum(0, np.outer(Xg[:, t], Wi) + H @ Wh.T); acts.append(H)
s = 1 / np.sqrt(np.mean(np.concatenate(acts) ** 2, 0))
Wi_g, Wh_g, Wo_g = Wi * s, (Wh * s[:, None]) / s[None, :], Wo / s[None, :]
X_ev = sample(921, 200_000); y_ev = pred(X_ev, (Wi, Wh, Wo))
assert (pred(X_ev, (Wi_g, Wh_g, Wo_g)) == y_ev).all()

T = lambda a: torch.tensor(a, dtype=torch.float32)
teach = [T(Wi_g), T(Wh_g), T(Wo_g)]


def fwd(X, wi, wh, wo):
    h = torch.zeros(X.shape[0], 16)
    for t in range(10):
        h = torch.relu(X[:, t:t + 1] * wi + h @ wh.T)
    return h @ wo.T


def run(r, k, steps, cfg):
    U, S, Vt = np.linalg.svd(Wh_g)
    u = T(U[:, :r] * np.sqrt(S[:r])).requires_grad_(); v = T(Vt[:r].T * np.sqrt(S[:r])).requires_grad_()
    m = Wo_g.mean(0); Ub, Sb, Vbt = np.linalg.svd(Wo_g - m)
    b = T(Ub[:, :k] * np.sqrt(Sb[:k])).requires_grad_(); c = T(Vbt[:k].T * np.sqrt(Sb[:k])).requires_grad_()
    wi = T(np.sign(Wi_g) * np.abs(Wi_g))  # kept exact in gauge; counted as 16 numbers below
    wi.requires_grad_()
    ps = [u, v, b, c, wi]
    opt = torch.optim.Adam(ps, lr=3e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    g = torch.Generator().manual_seed(920 + cfg)
    def build():
        return wi, u @ v.T, b @ c.T
    def ev():
        with torch.no_grad():
            w = build()
            p = fwd(T(X_ev), *w).argmax(1).numpy()
        return float((p == y_ev).mean())
    a0 = ev()
    for it in range(steps):
        X = torch.randn(4096, 10, generator=g)
        with torch.no_grad():
            pt = torch.log_softmax(fwd(X, *teach), 1)
        lq = torch.log_softmax(fwd(X, *build()), 1)
        loss = (pt.exp() * (pt - lq)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    free = (32 * r - r * r) + (26 * k - k * k)
    return {"r": r, "k": k, "free_numbers": free, "agree_svd_init": a0, "agree_ft": ev(), "steps": steps}


rows = []
cfgs = [(4, 4), (5, 2), (3, 5), (14, 9)]  # last = near-exact control (rank 14 W_hh, full W_oh): 405 numbers
for i, (r, k) in enumerate(cfgs):
    t0 = time.time(); row = run(r, k, 3000, i); row["sec"] = round(time.time() - t0, 1)
    rows.append(row); print(row, flush=True)
    json.dump(rows, open(f"{HERE}/skeptic/vera_lowrank.json", "w"), indent=1)
