"""Two-layer quadratic network (sigma(x)=x^2) of Cullen et al. (arXiv 2603.01192), in numpy float64.

f(x) = V sigma(W^T x),  W in R^{d x K},  V in R^{p x K}.
Loss on a data set (X, Y), n rows:  L_n = (1/2n) || P_perp (Y - X->f) ||_F^2,
P_perp = I - 11^T/n centres along the sample axis (the paper's centred squared loss).
Training adds L2 weight decay. The LLC is estimated on L_n alone (no L2 term).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- data
def modadd_data(p: int, frac: float, seed: int, return_idx: bool = False):
    """One-hot modular addition. Returns X_tr, Y_tr, X_te, Y_te (rows = samples) [, train_idx, test_idx].

    Row index i of the full table encodes (a, b) = (i // p, i % p).
    """
    rng = np.random.default_rng(seed)
    X, Y = full_table(p)
    perm = rng.permutation(p * p)
    n_tr = int(round(frac * p * p))
    tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
    if return_idx:
        return X[tr], Y[tr], X[te], Y[te], tr, te
    return X[tr], Y[tr], X[te], Y[te]


def full_table(p: int):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    c = (a + b) % p
    X = np.zeros((p * p, 2 * p))
    X[np.arange(p * p), a] = 1.0
    X[np.arange(p * p), p + b] = 1.0
    Y = np.eye(p)[c]
    return X, Y


# ---------------------------------------------------------------- neuron statistics
def neuron_stats(W, V, rel=0.01):
    """Per-neuron magnitude |v_j| |w_j|^2, Fourier purity and dominant frequency.

    Active (counted in K_eff) if magnitude >= rel * max. Purity = share of non-DC power of the
    two input halves (u_j = W[:p, j], s_j = W[p:, j]) in the strongest frequency pair {k, p-k}.
    """
    p = V.shape[0]
    mag = np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0) ** 2
    U = np.fft.rfft(W[:p], axis=0)[1:]
    S = np.fft.rfft(W[p:], axis=0)[1:]
    P = np.abs(U) ** 2 + np.abs(S) ** 2
    purity = P.max(0) / np.maximum(P.sum(0), 1e-300)
    freq = np.argmax(P, 0) + 1
    active = mag >= rel * mag.max()
    return dict(mag=mag, purity=purity, freq=freq, active=active, K_eff=int(active.sum()))


# ---------------------------------------------------------------- model
def forward(W, V, X):
    H = X @ W
    return (H * H) @ V.T, H


def loss(W, V, X, Y):
    Yh, _ = forward(W, V, X)
    R = Yh - Y
    R = R - R.mean(axis=0, keepdims=True)
    return 0.5 * np.sum(R * R) / X.shape[0]


def loss_grad(W, V, X, Y):
    n = X.shape[0]
    H = X @ W
    F = H * H
    R = F @ V.T - Y
    R = R - R.mean(axis=0, keepdims=True)
    L = 0.5 * np.sum(R * R) / n
    G = R / n                      # dL/dYhat (R already centred, P_perp is a projector)
    gV = G.T @ F                   # p x K
    gH = 2.0 * H * (G @ V)         # n x K
    gW = X.T @ gH                  # d x K
    return L, gW, gV


def accuracy(W, V, X, Y):
    return float(np.mean(np.argmax(forward(W, V, X)[0], 1) == np.argmax(Y, 1)))


# ---------------------------------------------------------------- Jacobian rank (diagnostic)
def jacobian(W, V, X):
    """Jacobian of the centred training outputs vec(P_perp Yhat) w.r.t. (W, V). Shape (n*p, dK + pK)."""
    n, d = X.shape
    p, K = V.shape
    H = X @ W
    F = H * H
    # d Yhat[i,o] / d W[j,k] = 2 H[i,k] V[o,k] X[i,j]
    JW = 2.0 * np.einsum("ik,ok,ij->iojk", H, V, X).reshape(n * p, d * K)
    # d Yhat[i,o] / d V[q,k] = delta(o,q) F[i,k]
    JV = np.einsum("ik,oq->ioqk", F, np.eye(p)).reshape(n * p, p * K)
    J = np.concatenate([JW, JV], axis=1).reshape(n, p, -1)
    J = J - J.mean(axis=0, keepdims=True)
    return J.reshape(n * p, -1)


def jac_rank(W, V, X, rtol=1e-6):
    s = np.linalg.svd(jacobian(W, V, X), compute_uv=False)
    return int(np.sum(s > rtol * s[0])), s


# ---------------------------------------------------------------- SGLD LLC (Lau et al., arXiv 2308.12108)
def sgld_llc(W0, V0, X, Y, eps, gamma, n_steps, n_burn, n_chains, seed, beta=None, thin=10,
             nbeta=None, freeze_W=False):
    """Full-batch SGLD around w* = (W0, V0) with localisation gamma.

    theta <- theta + eps/2 * ( -n beta grad L_n(theta) - gamma (theta - theta*) ) + N(0, eps)
    lambda_hat = n beta ( mean_{t >= burn} L_n(theta_t) - L_n(theta*) ).
    n beta defaults to n / log n (Lau et al.); pass nbeta to set it directly (it then overrides beta).
    freeze_W=True samples V only (the frozen-feature model of Thm 5.2).
    Returns dict(llc, chains, diverged, trace[chains, steps/thin], L0, nbeta).
    """
    n = X.shape[0]
    if nbeta is not None:
        nb = float(nbeta)
    else:
        nb = n * (1.0 / np.log(n) if beta is None else beta)
    L0 = loss(W0, V0, X, Y)
    rng = np.random.default_rng(seed)
    chains, traces = [], []
    diverged = False
    for c in range(n_chains):
        W, V = W0.copy(), V0.copy()
        Ls = np.empty(n_steps)
        for t in range(n_steps):
            L, gW, gV = loss_grad(W, V, X, Y)
            Ls[t] = L
            if not np.isfinite(L) or L > 1e6:
                diverged = True
                break
            if not freeze_W:
                W = W + 0.5 * eps * (-nb * gW - gamma * (W - W0)) + np.sqrt(eps) * rng.standard_normal(W.shape)
            V = V + 0.5 * eps * (-nb * gV - gamma * (V - V0)) + np.sqrt(eps) * rng.standard_normal(V.shape)
        if diverged:
            break
        chains.append(float(nb * (Ls[n_burn:].mean() - L0)))
        traces.append(Ls[::thin])
    out = dict(L0=float(L0), nbeta=float(nb), diverged=diverged)
    if diverged:
        out.update(llc=float("nan"), chains=[], trace=None)
    else:
        out.update(llc=float(np.mean(chains)), chains=chains, trace=np.array(traces))
    return out
