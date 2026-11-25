"""Vera's independent core for the rebuild (own code; does not import room code).
- aoyagi split: lambda, regular part r(M+N-r)/2, null block lambda0 = lambda(M-r,N-r,H-r,0)
- law predictions L_q (psi = 1/(1+(a/q)^b)) and L_z (Phi(z) = z/(1+z^b)^(1/b)), regular term closed form
  sum_i 1/2 * [1/(1+g_i)] / [1 - c_i/4], g_i = gamma/(nbeta h_i), c_i = eps (nbeta h_i + gamma)
- exact block Gibbs on the localised posterior with data covariance S (K = 1/2 tr((BA-W0) S (BA-W0)^T))
- minibatch SGLD on a noise-free teacher (y = W0 x), started at the special zero-padded w*
"""
import numpy as np
import sympy as sp
from vera_aoyagi import aoyagi


def lam_parts(M, N, H, r):
    lam = float(aoyagi(M, N, H, r)[0])
    reg = r * (M + N - r) / 2
    lam0 = float(aoyagi(M - r, N - r, H - r, 0)[0]) if H > r else 0.0
    assert abs(lam - reg - lam0) < 1e-9
    return lam, reg, lam0


def teacher(M, N, H, r, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, r)))
    V, _ = np.linalg.qr(rng.standard_normal((M, r)))
    W0 = U @ V.T
    B0 = np.zeros((N, H)); A0 = np.zeros((H, M))
    B0[:, :r] = U; A0[:r] = V.T
    return W0, B0, A0


def hess_eigs(B0, A0, S):
    """Nonzero eigenvalues of the Gauss-Newton Hessian of K = 1/2 tr(D S D^T) at w* (D = BA - W0 = 0)."""
    N, H = B0.shape; M = A0.shape[1]
    Ls = np.linalg.cholesky(S)  # K = 1/2 ||D Ls||^2
    cols = []
    for i in range(N):
        for k in range(H):
            E = np.zeros((N, H)); E[i, k] = 1; cols.append((E @ A0 @ Ls).ravel())
    for k in range(H):
        for j in range(M):
            E = np.zeros((H, M)); E[k, j] = 1; cols.append((B0 @ E @ Ls).ravel())
    J = np.array(cols).T
    ev = np.linalg.eigvalsh(J.T @ J)
    return ev[ev > 1e-8 * ev.max()]


def reg_term(h, nbeta, gamma, eps):
    g = gamma / (nbeta * h)
    c = eps * (nbeta * h + gamma)
    return float(np.sum(0.5 / (1 + g) / (1 - c / 4)))


def psi_q(q, a, b):
    return 1.0 / (1.0 + (a / q) ** b)


def phi_z(z, b):
    return z / (1.0 + z ** b) ** (1.0 / b)


def gibbs(W0, B0, A0, S, gamma, nbeta, sweeps, burn, rng):
    N, H = B0.shape; M = A0.shape[1]
    I = np.eye(H)
    B, A = B0.copy(), A0.copy()
    acc, cnt = 0.0, 0
    # A | B: vec over columns couples through S; use eigenbasis of S: S = Q diag(s) Q^T, A~ = A Q
    s, Q = np.linalg.eigh(S)
    W0t = W0 @ Q; A0t = A0 @ Q
    for t in range(sweeps):
        # B | A: rows independent, precision nbeta A S A^T + gamma I
        P = nbeta * A @ S @ A.T + gamma * I
        L = np.linalg.cholesky(P)
        mean = np.linalg.solve(P, (nbeta * W0 @ S @ A.T + gamma * B0).T).T
        B = mean + np.linalg.solve(L.T, rng.standard_normal((H, N))).T
        # A~ | B: column j of A~ has precision nbeta s_j B^T B + gamma I
        BtB = B.T @ B; BtW = B.T @ W0t
        At = np.empty((H, M))
        e, R = np.linalg.eigh(BtB)
        for j in range(M):
            d = nbeta * s[j] * e + gamma
            rhs = R.T @ (nbeta * s[j] * BtW[:, j] + gamma * A0t[:, j])
            At[:, j] = R @ (rhs / d + rng.standard_normal(H) / np.sqrt(d))
        A = At @ Q.T
        if t >= burn:
            D = B @ A - W0
            acc += 0.5 * np.trace(D @ S @ D.T); cnt += 1
    return nbeta * acc / cnt


def sgld(W0, B0, A0, X, gamma, nbeta, eps, T, batch, rng, burn_frac=0.25, chains=1):
    """Minibatch SGLD, noise-free teacher; loss per sample 1/2||(BA - W0) x||^2; returns nbeta*mean K_full over draws."""
    n = X.shape[0]
    S = X.T @ X / n
    B = np.repeat(B0[None], chains, 0); A = np.repeat(A0[None], chains, 0)
    burn = int(T * burn_frac); acc = np.zeros(chains); cnt = 0
    for t in range(T):
        idx = rng.integers(0, n, batch)
        Sb = X[idx].T @ X[idx] / batch
        D = B @ A - W0
        G = D @ Sb  # dL/d(BA)
        gB = G @ A.transpose(0, 2, 1); gA = B.transpose(0, 2, 1) @ G
        B = B - eps / 2 * (nbeta * gB + gamma * (B - B0)) + np.sqrt(eps) * rng.standard_normal(B.shape)
        A = A - eps / 2 * (nbeta * gA + gamma * (A - A0)) + np.sqrt(eps) * rng.standard_normal(A.shape)
        if t >= burn:
            Df = B @ A - W0
            acc += 0.5 * np.einsum('cij,jk,cik->c', Df, S, Df); cnt += 1
    return nbeta * acc / cnt
