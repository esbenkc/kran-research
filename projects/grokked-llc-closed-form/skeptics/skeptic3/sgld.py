"""skeptic3: independent SGLD LLC estimator (numpy, float64, full batch).

Written from Lau et al. (arXiv 2308.12108), not from the build code.
Model: f_c(a,b) = sum_j V[c,j] (U[j,a] + S[j,b])^2.  W (2p x K) of the paper = [U^T; S^T].
L_n(theta) = (1/(2N)) * || P_perp (Y - f) ||_F^2 over the N train pairs (P_perp centres each
output over the samples).  No L2 term.
SGLD:  theta <- theta - (eps/2) (nbeta grad L_n + gamma (theta - theta*)) + N(0, eps)
lambda_hat = nbeta * (mean_{post-burn} L_n - L_n(theta*)).
freeze_W=True: only V moves (the random-feature model of Thm 5.2).
"""
import numpy as np


def forward(U, S, V, A, B):
    H = U[:, A] + S[:, B]  # K x N
    return V @ (H ** 2), H  # p x N


def centred_residual(U, S, V, A, B, Y):
    F, H = forward(U, S, V, A, B)
    R = Y - F
    return R - R.mean(axis=1, keepdims=True), H


def loss(U, S, V, A, B, Y):
    R, _ = centred_residual(U, S, V, A, B, Y)
    return 0.5 * (R ** 2).sum() / R.shape[1]


def grad(U, S, V, A, B, Y, OA=None, OB=None):
    R, H = centred_residual(U, S, V, A, B, Y)
    N = R.shape[1]
    G = -R / N  # dL/dF  (p x N), already centred
    gV = G @ (H ** 2).T  # p x K
    gH = 2 * H * (V.T @ G)  # K x N
    p = U.shape[1]
    if OA is None:
        OA, OB = np.eye(p)[A], np.eye(p)[B]  # N x p one-hot scatter matrices
    return gH @ OA, gH @ OB, gV, 0.5 * (R ** 2).sum() / N


def llc(U0, S0, V0, A, B, Y, nbeta, eps, gamma, steps, burn, chains, seed, freeze_W=False):
    rng = np.random.default_rng(seed)
    L0 = loss(U0, S0, V0, A, B, Y)
    OA, OB = np.eye(U0.shape[1])[A], np.eye(U0.shape[1])[B]
    out = []
    traces = []
    for ch in range(chains):
        U, S, V = U0.copy(), S0.copy(), V0.copy()
        Ls = []
        for t in range(steps):
            gU, gS, gV, L = grad(U, S, V, A, B, Y, OA, OB)
            if t >= burn:
                Ls.append(L)
            se = np.sqrt(eps)
            V = V - 0.5 * eps * (nbeta * gV + gamma * (V - V0)) + se * rng.standard_normal(V.shape)
            if not freeze_W:
                U = U - 0.5 * eps * (nbeta * gU + gamma * (U - U0)) + se * rng.standard_normal(U.shape)
                S = S - 0.5 * eps * (nbeta * gS + gamma * (S - S0)) + se * rng.standard_normal(S.shape)
            if not np.isfinite(L):
                break
        Ls = np.array(Ls)
        out.append(nbeta * (Ls.mean() - L0) if len(Ls) and np.isfinite(Ls).all() else np.nan)
        traces.append(Ls[:: max(1, len(Ls) // 50)].tolist())
    out = np.array(out)
    return dict(lambda_hat=float(np.nanmean(out)), chain_se=float(np.nanstd(out, ddof=1) / np.sqrt(len(out))) if chains > 1 else float("nan"),
                per_chain=out.tolist(), L0=float(L0))
