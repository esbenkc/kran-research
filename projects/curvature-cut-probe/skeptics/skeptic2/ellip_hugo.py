"""Hugo (skeptic2): found-move cost on the RMSNorm ellipsoid, per eval twin pair, own code.

Per state (post-norm z0 at the country token, p0 = softmax(W z0)):
  frontier by Lagrangian tilting on the sphere y = z/gamma, ||y|| = ||y0||:
      min_y KL(p0 || softmax(W gamma*y)) + lam * sign * (s(z) - s(z0)),  lam on a log grid
  sign +1: false state pushed down; sign -1: true twin pushed up. Every solution is a real point on
  the ellipsoid; its EXACT full-vocab KL is re-evaluated (streamed W) after the search.
  c(a) = min exact KL over found points with shift >= a  (upper bound on the reachable minimum).
Search uses a vocab subset (union of tokens with p0 > 1e-7 over the batch, plus the top 4096 of the mean
p0); the final cost is always the exact full-vocab KL, so truncation can only make the bound looser.
Pair cost (D5): min over a in {0, D/4, D/2, 3D/4, D} of c_false(a) + c_true(D - a), D = s_false - line_clean.
Second-order pair formula: D^2 / (2 (g_false + g_true)), g = u'(F + eps I)^-1 u from gate_hugo.py.

usage: ellip_hugo.py NPAIRS ARM [ARM...]   -> out/ellip_<arm>.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

from gate_hugo import DATA, HERE, chunks, logits, pairs_order, V, d

torch.set_num_threads(4)


def exact_kl_many(lp0_full, Zp):
    """KL(p0_i||softmax(W z_i)) over the full vocab for each row, W streamed once."""
    L = np.empty((len(Zp), V))
    for a, Wc in chunks():
        L[:, a:a + len(Wc)] = Zp @ Wc.astype(np.float64).T
    L -= L.max(1, keepdims=True)
    lq = L - np.log(np.exp(L).sum(1, keepdims=True))
    p0 = np.exp(lp0_full)
    return (p0 * (lp0_full - lq)).sum(1)


def frontier(Wsub, lp0, y0, g, u, sign, lams, steps, lr, polish=100):
    """Batched tilt on the sphere: Adam warm-up then L-BFGS polish per lam (problems are separable,
    the summed loss only shares the line search). Returns list of z points per lam."""
    p0 = lp0.exp()
    r = y0.norm(dim=-1, keepdim=True)
    s0 = (g * y0) @ u
    v = y0.clone().requires_grad_(True)
    pts = []

    def lossf(lam):
        z = g * (r * v / v.norm(dim=-1, keepdim=True))
        lq = torch.log_softmax(z @ Wsub.T, -1)
        kl = (p0 * (lp0 - lq)).sum(-1)
        return (kl + lam * sign * (z @ u - s0)).sum()

    for lam in lams:
        opt = torch.optim.Adam([v], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-3)
        for _ in range(steps):
            opt.zero_grad()
            loss = lossf(lam)
            loss.backward()
            opt.step()
            sch.step()
        if polish:
            lb = torch.optim.LBFGS([v], lr=1, max_iter=polish, history_size=30, line_search_fn='strong_wolfe',
                                   tolerance_grad=1e-15, tolerance_change=1e-18)

            def closure():
                lb.zero_grad()
                loss = lossf(lam)
                loss.backward()
                return loss
            lb.step(closure)
        with torch.no_grad():
            pts.append((g * (r * v / v.norm(dim=-1, keepdim=True))).clone())
    return pts


def main():
    npairs = int(sys.argv[1])
    arms = sys.argv[2:]
    steps = int(os.environ.get('STEPS', 200))
    lr = float(os.environ.get('LR', 0.02))
    off = int(os.environ.get('OFF', 0))
    lams = np.logspace(float(os.environ.get('LAMLO', -5)), float(os.environ.get('LAMHI', -1.5)),
                       int(os.environ.get('NLAM', 15)))
    split = json.load(open(os.path.join(DATA, 'ref/split.json')))
    zall = np.load(os.path.join(DATA, 'clean/z_post.npy'), mmap_mode='r')
    pz = np.load(os.path.join(DATA, 'fits/probes.npz'))
    from safetensors import safe_open
    from fisher_hugo import SNAP
    with safe_open(SNAP, 'pt') as s:
        gam = s.get_tensor('model.norm.weight').float().numpy().astype(np.float64)
    order = pairs_order(split)[off:off + npairs]
    idx = [i for pr in order for i in pr]  # false, true, false, true ...
    sign = np.array([1.0 if split[i]['true'] == 0 else -1.0 for i in idx])
    Z = np.asarray(zall[idx], dtype=np.float64)
    # full log p0 per state (streamed), and the search vocab
    lp0_full = []
    for zi in Z:
        l = logits(zi)
        l -= l.max()
        lp0_full.append(l - np.log(np.exp(l).sum()))
    lp0_full = np.array(lp0_full)
    keep = np.nonzero((lp0_full > np.log(1e-7)).any(0))[0]
    meanp = np.exp(lp0_full).mean(0)
    keep = np.union1d(keep, np.argsort(meanp)[::-1][:4096])
    Wsub = np.empty((len(keep), d), dtype=np.float32)
    pos = 0
    for a, Wc in chunks():
        m = (keep >= a) & (keep < a + len(Wc))
        k = keep[m] - a
        Wsub[pos:pos + len(k)] = Wc[k]
        pos += len(k)
    Wsub_t = torch.from_numpy(Wsub.astype(np.float64))
    del Wsub
    lp0_sub = torch.log_softmax(torch.from_numpy(Z) @ Wsub_t.T, -1)
    out = {'npairs': npairs, 'nvocab_search': int(len(keep)), 'steps': steps, 'lr': lr, 'lams': lams.tolist(),
           'idx': idx, 'arms': {}}
    g = torch.from_numpy(gam)
    y0 = torch.from_numpy(Z) / g
    sg = torch.from_numpy(sign)
    for arm in arms:
        t0 = time.time()
        w = pz[f'{arm}__w'].astype(np.float64)
        sd = pz[f'{arm}__sd'].astype(np.float64)
        mu = pz[f'{arm}__mu'].astype(np.float64)
        b = float(pz[f'{arm}__b'])
        u64 = w / sd
        sc_all = ((np.asarray(zall, dtype=np.float64) - mu) / sd) @ w + b
        tr = np.array([x['true'] for x in split])
        sp = np.array([x['split'] for x in split])
        line = np.sort(sc_all[(sp == 'refit') & (tr == 1)])[::-1][1]
        s0 = sc_all[idx]
        pts = frontier(Wsub_t, lp0_sub, y0, g, torch.from_numpy(u64), sg, lams, steps, lr,
                       polish=int(os.environ.get('POLISH', 100)))
        shifts = np.zeros((len(idx), len(lams)))
        kls = np.zeros((len(idx), len(lams)))
        allP = []
        for j, P in enumerate(pts):
            P = P.numpy().astype(np.float64)
            allP.append(P)
            shifts[:, j] = sign * (s0 - (((P - mu) / sd) @ w + b))
            kls[:, j] = np.clip(exact_kl_many(lp0_full, P), 0.0, None)
        out['arms'][arm] = dict(line=float(line), s0=s0.tolist(), shifts=shifts.tolist(), kls=kls.tolist(),
                                sec=time.time() - t0)
        tag = f'{"_".join(arms)}_off{off}_n{npairs}'
        np.savez_compressed(os.path.join(HERE, 'out', f'ellip_pts_{arm}_off{off}_n{npairs}.npz'),
                            idx=np.array(idx), z=np.array(allP).astype(np.float32), lams=lams)
        print(arm, 'done', round(time.time() - t0), 's', flush=True)
        json.dump(out, open(os.path.join(HERE, 'out', f'ellip_{tag}.json'), 'w'))


if __name__ == '__main__':
    main()
