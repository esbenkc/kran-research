"""The ONE SGLD local-learning-coefficient estimator, shared by the gate (gate.py, LoRA-shaped reduced-rank regression)
and the 8B tail adapter (llc.py). Lau et al., arXiv 2308.12108 (eq. 9 + the localised SGLD of their section 4):

    w_{t+1} = w_t - (eps/2) * ( n*beta * grad L_S(w_t) + gamma * (w_t - w*) ) + N(0, eps I)
    lambda_hat = n*beta * ( mean_{draws} L_S(w_t) - L_n(w*) ),        n*beta = n / ln n

L_S = the loss on a random minibatch S (|S| = batch rows, drawn without replacement per step), L_n the loss on all n
rows. Here L_n(w*) = 0 exactly in both uses (noise-free RRR at the teacher; KL(pi_ckpt || pi_w) at w = ckpt).
Step size rule (dimensionless, frozen by the gate): eps = c / (n*beta * h_max + gamma), h_max = top eigenvalue of the
Hessian of L_n at w*, from power iteration with central finite-difference Hessian-vector products (hmax_power).
"""
from __future__ import annotations

import math

import torch


def nbeta_of(n: int) -> float:
    return n / math.log(n)


def step_size(c: float, nbeta: float, h_max: float, gamma: float) -> float:
    return c / (nbeta * h_max + gamma)


def _flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def hmax_power(grad_fn, params, iters: int = 8, delta: float = 1e-2, seed: int = 0):
    """Top Hessian eigenvalue of the loss behind grad_fn at the current params (power iteration, central differences).
    grad_fn() -> list of grads (same shapes as params) of a FIXED loss. Params are restored exactly afterwards."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w0 = [p.detach().clone() for p in params]
    v = [torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for p in params]
    nv = float(torch.sqrt(sum((x.double() ** 2).sum() for x in v)))
    v = [x / nv for x in v]
    lam = float("nan")
    hist = []
    for _ in range(iters):
        with torch.no_grad():
            for p, a, x in zip(params, w0, v):
                p.copy_(a + delta * x)
        gp = grad_fn()
        with torch.no_grad():
            for p, a, x in zip(params, w0, v):
                p.copy_(a - delta * x)
        gm = grad_fn()
        hv = [(a - b) / (2 * delta) for a, b in zip(gp, gm)]
        lam = float(sum((a.double() * x.double()).sum() for a, x in zip(hv, v)))  # Rayleigh quotient v.Hv, |v| = 1
        nh = float(torch.sqrt(sum((x.double() ** 2).sum() for x in hv)))
        hist.append(lam)
        if nh == 0:
            break
        v = [x / nh for x in hv]
    with torch.no_grad():
        for p, a in zip(params, w0):
            p.copy_(a)
    return lam, hist


def sgld_chain(loss_fn, params, w_star, n: int, batch: int, nbeta: float, eps: float, gamma: float, burn: int,
               draws: int, seed: int, full_loss_fn=None, mala: bool = False, end_fn=None):
    """One localised SGLD chain started at w*. loss_fn(rows) -> scalar loss on those row indices (differentiable in
    params). Returns per-step minibatch losses, lambda_hat over the draws, and (if mala) MALA acceptance probabilities
    computed with full_loss_fn on the full data (diagnostic only; nothing is rejected)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p, a in zip(params, w_star):
            p.copy_(a)
    losses, accs, dist = [], [], []
    perm, pos = torch.randperm(n, generator=g), 0

    def U_and_grad(lfn):
        l = lfn()
        gr = torch.autograd.grad(l, params)
        loc = sum(((p - a).double() ** 2).sum() for p, a in zip(params, w_star))
        U = nbeta * float(l) + 0.5 * gamma * float(loc)
        dU = [nbeta * gg + gamma * (p.detach() - a) for gg, p, a in zip(gr, params, w_star)]
        return U, dU

    for t in range(burn + draws):
        if pos + batch > n:
            perm, pos = torch.randperm(n, generator=g), 0
        rows = perm[pos:pos + batch].tolist()
        pos += batch
        l = loss_fn(rows)
        gr = torch.autograd.grad(l, params)
        losses.append(float(l.detach()))
        if mala:
            U0, dU0 = U_and_grad(full_loss_fn)
            w_old = [p.detach().clone() for p in params]
        noise = [torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for p in params]
        with torch.no_grad():
            for p, gg, a, z in zip(params, gr, w_star, noise):
                p.add_(-(eps / 2) * (nbeta * gg + gamma * (p - a)) + math.sqrt(eps) * z)
        if mala:
            U1, dU1 = U_and_grad(full_loss_fn)
            fwd = sum(((p.detach() - wo + (eps / 2) * d0).double() ** 2).sum() for p, wo, d0 in zip(params, w_old, dU0))
            bwd = sum(((wo - p.detach() + (eps / 2) * d1).double() ** 2).sum() for p, wo, d1 in zip(params, w_old, dU1))
            log_a = -U1 + U0 - float(bwd) / (2 * eps) + float(fwd) / (2 * eps)
            accs.append(min(1.0, math.exp(min(0.0, log_a))))
        if t % 25 == 24 or t == burn + draws - 1:
            dist.append(float(torch.sqrt(sum(((p.detach() - a).double() ** 2).sum() for p, a in zip(params, w_star)))))
    lam = nbeta * (sum(losses[burn:]) / draws)
    end = end_fn() if end_fn is not None else None  # e.g. the full-batch loss at the last draw (before the reset)
    with torch.no_grad():
        for p, a in zip(params, w_star):
            p.copy_(a)
    return {"lambda_hat": lam, "losses": losses, "mala_accept": accs, "dist_from_wstar": dist, "end": end}
