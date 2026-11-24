"""Vera's own SGLD LLC on a LoRA-shaped reduced-rank regression, to rebuild the Phase-0 gate independently.
y = B0 A0 x + s*eps, x ~ N(0, I_M), model y = B A x, B: N x H, A: H x M.
Loss (per-sample Gaussian NLL minus const): l(w; x, y) = ||y - B A x||^2 / (2 s^2); L_n = mean over n.
SGLD (Lau et al. 2308.12108): w <- w - (eps/2) (n beta grad L_batch(w) + gamma (w - w*)) + N(0, eps I)
LLC-hat = n beta (mean over draws of L_batch(w) - L_n(w*)), n beta = n / ln n.
All settings come from the command line so the frozen values can be plugged in after they are posted."""
import argparse, json, math
import numpy as np
import torch
from vera_aoyagi import aoyagi

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=32); ap.add_argument("--N", type=int, default=32)
ap.add_argument("--H", type=int, default=16); ap.add_argument("--r", type=int, nargs="+", default=[1, 8])
ap.add_argument("--n", type=int, required=True); ap.add_argument("--s", type=float, default=1.0)
ap.add_argument("--c", type=float, required=True); ap.add_argument("--gamma", type=float, required=True)
ap.add_argument("--chains", type=int, default=2); ap.add_argument("--burn", type=int, default=25)
ap.add_argument("--draws", type=int, default=75); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--data-seed", type=int, default=0); ap.add_argument("--sgld-seed", type=int, default=100)
ap.add_argument("--wstar", choices=["true", "fit"], default="fit")
ap.add_argument("--eps", type=float, default=None, help="override eps directly")
ap.add_argument("--sv", type=float, default=1.0, help="true singular values")
ap.add_argument("--out", default=None)
ap.add_argument("--loss", choices=["nll", "kl"], default="kl", help="kl: ||(BA - B*A*)x||^2/(2 s^2), w* = truth")
a = ap.parse_args()
torch.set_default_dtype(torch.float64)


def hess_top(fn, w, iters=200):
    v = torch.randn_like(w); v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        g = torch.autograd.grad(fn(w), w, create_graph=True)[0]
        hv = torch.autograd.grad(g @ v, w)[0]
        lam = (hv @ v).item(); v = hv / hv.norm()
    return lam


res = {}
for r in a.r:
    rng = np.random.default_rng(a.data_seed + 1000 * r)
    U = np.linalg.qr(rng.standard_normal((a.N, r)))[0]; V = np.linalg.qr(rng.standard_normal((a.M, r)))[0]
    W0 = torch.tensor(U @ np.diag([a.sv] * r) @ V.T)
    X = torch.tensor(rng.standard_normal((a.n, a.M)))
    Y = X @ W0.T + a.s * torch.tensor(rng.standard_normal((a.n, a.N)))
    dA, dB = a.H * a.M, a.N * a.H

    def unpack(w):
        return w[dA:].view(a.N, a.H), w[:dA].view(a.H, a.M)

    def L(w, idx=None):
        B, A = unpack(w)
        x, y = (X, Y) if idx is None else (X[idx], Y[idx])
        if a.loss == "kl":
            y = x @ W0.T
        return ((y - x @ (B @ A).T) ** 2).sum(-1).mean() / (2 * a.s ** 2)

    # w*: factor the truth (padded with zeros) or fit by full-batch descent from the truth
    Ut, St, Vt = torch.linalg.svd(W0)
    B0 = torch.zeros(a.N, a.H); A0 = torch.zeros(a.H, a.M)
    B0[:, :r] = Ut[:, :r] * St[:r].sqrt(); A0[:r] = St[:r, None].sqrt() * Vt[:r]
    w = torch.cat([A0.flatten(), B0.flatten()]).requires_grad_(True)
    if a.wstar == "fit" and a.loss == "nll":
        opt = torch.optim.Adam([w], lr=1e-3)
        for i in range(20000):
            opt.zero_grad(); l = L(w); l.backward(); opt.step()
            if w.grad.norm() < 1e-6: break
    wstar = w.detach().clone()
    Lstar = L(wstar).item()
    nb = a.n / math.log(a.n)
    hmax = hess_top(lambda z: L(z), wstar.clone().requires_grad_(True))
    eps = a.eps if a.eps is not None else a.c / (nb * hmax + a.gamma)
    g = torch.Generator().manual_seed(a.sgld_seed + r)
    draws = []
    for ch in range(a.chains):
        w = wstar.clone().requires_grad_(True)
        for step in range(a.burn + a.draws):
            idx = torch.randint(0, a.n, (a.batch,), generator=g)
            lb = L(w, idx)
            gr = torch.autograd.grad(lb, w)[0]
            with torch.no_grad():
                w -= (eps / 2) * (nb * gr + a.gamma * (w - wstar))
                w += math.sqrt(eps) * torch.randn(w.shape, generator=g)
            if step >= a.burn:
                draws.append((ch, lb.item()))
    d = np.array([x[1] for x in draws])
    llc = nb * (d.mean() - Lstar)
    bm = np.array([nb * (b.mean() - Lstar) for b in np.array_split(d, 5)])
    se = bm.std(ddof=1) / math.sqrt(len(bm))
    lam, m = aoyagi(a.M, a.N, a.H, r)
    res[r] = dict(llc=llc, se=se, aoyagi=float(lam), mult=m, rel_err=llc / float(lam) - 1, eps=eps, hmax=hmax,
                  nbeta=nb, Lstar=Lstar)
    print(f"r={r}: LLC-hat {llc:.2f} (batch-means SE {se:.2f})  Aoyagi {float(lam)} (m={m})  rel err {llc/float(lam)-1:+.3f}  eps {eps:.3g} hmax {hmax:.3g}")
if len(a.r) == 2:
    r1, r2 = a.r
    sep = abs(res[r2]["llc"] - res[r1]["llc"]) / max(res[r1]["se"], res[r2]["se"])
    ok = all(abs(res[r]["rel_err"]) < 0.10 for r in a.r) and sep >= 3
    print(f"separation {sep:.1f} SE; gate pass = {ok}")
    res["sep"] = sep; res["pass"] = ok
if a.out:
    json.dump({str(k): v for k, v in res.items()}, open(a.out, "w"), indent=1)
