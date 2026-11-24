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


def _dot(a, b) -> float:
    """sum(a * b) as a Python float; fp64 on CPU, fp32 on MPS (MPS has no float64)."""
    if a.device.type == "mps":
        return float((a.float() * b.float()).sum())
    return float((a.double() * b.double()).sum())


def hmax_power(grad_fn, params, iters: int = 8, delta: float = 1e-2, seed: int = 0):
    """Top Hessian eigenvalue of the loss behind grad_fn at the current params (power iteration, central differences).
    grad_fn() -> list of grads (same shapes as params) of a FIXED loss. Params are restored exactly afterwards."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w0 = [p.detach().clone() for p in params]
    v = [torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for p in params]
    nv = sum(_dot(x, x) for x in v) ** 0.5
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
        lam = sum(_dot(a, x) for a, x in zip(hv, v))  # Rayleigh quotient v.Hv, |v| = 1
        nh = sum(_dot(x, x) for x in hv) ** 0.5
        hist.append(lam)
        if nh == 0:
            break
        v = [x / nh for x in hv]
    with torch.no_grad():
        for p, a in zip(params, w0):
            p.copy_(a)
    return lam, hist


def sgld_chain(loss_fn, params, w_star, n: int, batch: int, nbeta: float, eps: float, gamma: float, burn: int,
               draws: int, seed: int, full_loss_fn=None, mala: bool = False, end_fn=None,
               noise_gen=None):
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
        loc = sum(_dot(p - a, p - a) for p, a in zip(params, w_star))
        U = nbeta * float(l) + 0.5 * gamma * loc
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
        if noise_gen is None:  # CPU stream (gate, E2)
            noise = [torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for p in params]
        else:  # on-device stream (E2b, Vera #30068: no CPU->MPS staging per step)
            noise = [torch.randn(p.shape, generator=noise_gen, dtype=p.dtype, device=p.device) for p in params]
        with torch.no_grad():
            for p, gg, a, z in zip(params, gr, w_star, noise):
                p.add_(-(eps / 2) * (nbeta * gg + gamma * (p - a)) + math.sqrt(eps) * z)
        if mala:
            U1, dU1 = U_and_grad(full_loss_fn)
            fwd = sum(_dot(p.detach() - wo + (eps / 2) * d0, p.detach() - wo + (eps / 2) * d0) for p, wo, d0 in zip(params, w_old, dU0))
            bwd = sum(_dot(wo - p.detach() + (eps / 2) * d1, wo - p.detach() + (eps / 2) * d1) for p, wo, d1 in zip(params, w_old, dU1))
            log_a = -U1 + U0 - bwd / (2 * eps) + fwd / (2 * eps)
            accs.append(min(1.0, math.exp(min(0.0, log_a))))
        if t % 25 == 24 or t == burn + draws - 1:
            if params[0].device.type == "mps":  # release cached MPS blocks (memory cap; no effect on values)
                torch.mps.empty_cache()
            dist.append(sum(_dot(p.detach() - a, p.detach() - a) for p, a in zip(params, w_star)) ** 0.5)
    lam = nbeta * (sum(losses[burn:]) / draws)
    end = end_fn() if end_fn is not None else None  # e.g. the full-batch loss at the last draw (before the reset)
    with torch.no_grad():
        for p, a in zip(params, w_star):
            p.copy_(a)
    return {"lambda_hat": lam, "losses": losses, "mala_accept": accs, "dist_from_wstar": dist, "end": end}
