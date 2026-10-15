"""skeptic2: independent SGLD-LLC implementation (Lau et al., arXiv 2308.12108), numpy float64.

Written from the paper's definition, not from build's qnet.py:
  posterior  ~ exp(-n beta L_n(w) - gamma/2 |w - w*|^2)
  SGLD step  w <- w - (eps/2) (n beta grad L_n(w) + gamma (w - w*)) + sqrt(eps) xi
  estimate   lambda_hat = n beta ( E_t[L_n(w_t)] - L_n(w*) )
All chains run in parallel as a leading batch axis.

Model (one-hot pairs, data-level form, no X matmul):
  f_c(a,b) = sum_j V[c,j] (U[a,j] + S[b,j])^2 ,  W = [U; S] (2p x K)
  L_n = (1/(2N)) sum_n || centred residual_n ||^2   (centring over the N train samples)
"""
from __future__ import annotations

import numpy as np


def qnet_loss_grad(U, S, V, A, B, Y):
    """Batched over chains: U,S (C,p,K), V (C,p,K); A,B int (N,), Y (N,p).
    Returns L (C,), gU, gS, gV."""
    N = len(A)
    z = U[:, A, :] + S[:, B, :]  # (C,N,K)
    H = z * z
    R = np.einsum("cnk,cok->cno", H, V) - Y[None]  # (C,N,p)
    R = R - R.mean(axis=1, keepdims=True)
    L = 0.5 * np.einsum("cno,cno->c", R, R) / N
    G = R / N
    gV = np.einsum("cno,cnk->cok", G, H)
    gz = 2.0 * z * np.einsum("cno,cok->cnk", G, V)  # (C,N,K)
    p = U.shape[1]
    gU = np.zeros_like(U)
    gS = np.zeros_like(S)
    for c in range(U.shape[0]):
        np.add.at(gU[c], A, gz[c])
        np.add.at(gS[c], B, gz[c])
    return L, gU, gS, gV


def sgld_llc_qnet(W, V, A, B, Y, eps, gamma, nbeta, steps, burn, chains, seed=0, freeze_W=False):
    """W (2p,K), V (p,K). Returns dict(lambda_hat, se, per_chain, L0, diverged, trace_mean)."""
    p = V.shape[0]
    U0, S0, V0 = W[:p][None], W[p:][None], V[None]
    L0 = qnet_loss_grad(U0, S0, V0, A, B, Y)[0][0]
    rng = np.random.default_rng(seed)
    U = np.repeat(U0, chains, 0); S = np.repeat(S0, chains, 0); Vc = np.repeat(V0, chains, 0)
    acc = np.zeros(chains); cnt = 0
    trace = []
    for t in range(steps):
        L, gU, gS, gV = qnet_loss_grad(U, S, Vc, A, B, Y)
        if not np.all(np.isfinite(L)) or L.max() > 1e8:
            return dict(lambda_hat=float("nan"), se=float("nan"), per_chain=[], L0=float(L0), diverged=True)
        if t >= burn:
            acc += L; cnt += 1
        trace.append(float(L.mean()))
        h = 0.5 * eps
        Vc = Vc - h * (nbeta * gV + gamma * (Vc - V0)) + np.sqrt(eps) * rng.standard_normal(Vc.shape)
        if not freeze_W:
            U = U - h * (nbeta * gU + gamma * (U - U0)) + np.sqrt(eps) * rng.standard_normal(U.shape)
            S = S - h * (nbeta * gS + gamma * (S - S0)) + np.sqrt(eps) * rng.standard_normal(S.shape)
    per = nbeta * (acc / cnt - L0)
    return dict(lambda_hat=float(per.mean()), se=float(per.std(ddof=1) / np.sqrt(chains)) if chains > 1 else float("nan"),
                per_chain=per.tolist(), L0=float(L0), diverged=False, trace=trace)


def sgld_llc_quadratic(h, eps, gamma, nbeta, steps, burn, chains, seed=0):
    """Regular null model: L(w) = 1/2 sum_i h_i w_i^2 (w*=0), true LLC = d/2.
    Returns the SGLD estimate and the exact discrete-time stationary expectation."""
    rng = np.random.default_rng(seed)
    d = len(h)
    w = np.zeros((chains, d))
    acc = np.zeros(chains); cnt = 0
    for t in range(steps):
        if t >= burn:
            acc += 0.5 * (h * w * w).sum(1); cnt += 1
        w = w - 0.5 * eps * (nbeta * h + gamma) * w + np.sqrt(eps) * rng.standard_normal(w.shape)
    per = nbeta * acc / cnt
    a = nbeta * h + gamma
    var = 1.0 / (a * (1.0 - eps * a / 4.0))  # stationary variance of the Euler-Maruyama OU chain
    exact = float((nbeta * 0.5 * h * var).sum())
    continuum = float((nbeta * h / (2 * (nbeta * h + gamma))).sum())
    return dict(lambda_hat=float(per.mean()), se=float(per.std(ddof=1) / np.sqrt(chains)),
                exact_discrete=exact, exact_continuum=continuum, truth=d / 2, stable=bool((eps * a < 4).all()))
