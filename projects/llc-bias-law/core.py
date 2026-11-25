"""Batched localised SGLD + exact block Gibbs on noise-free reduced-rank regression (Bruno).

Model y = B A x, A: H x M, B: N x H. Teacher W0 = U V^T, rank r, singular values 1 (QR of Gaussians).
w* = the SPECIAL zero-padded point A = [V^T; 0], B = [U, 0] (frozen; local LC = Aoyagi's lambda only there).
Loss L(w) = (1/2) mean_i ||y_i - B A x_i||^2 = (1/2) tr((BA - W0) S (BA - W0)^T), S = X^T X / n; L(w*) = 0.

SGLD (Lau et al. 2308.12108, localised):  w <- w - (eps/2)(nbeta grad L_b(w) + gamma (w - w*)) + N(0, eps I)
  arm S: L_b = loss on a minibatch of `batch` = 64 rows per step per chain, drawn without replacement within an
  epoch (a fresh permutation per epoch); arm P: the exact population gradient of (1/2)||BA - W0||^2 (S = I).
lambda_hat = nbeta * mean_{t >= burn} K_n(w_t), K_n on ALL n rows (= (1/2) tr(D S D^T), D = BA - W0); arm P: K(w_t).
Every chain k carries its own data X_k, teacher, eps_k, gamma_k, nbeta_k. Losses are summed in blocks.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def teacher(M, N, H, r, seed):
    g = np.random.default_rng([seed, 0])
    U, _ = np.linalg.qr(g.standard_normal((N, r)))
    V, _ = np.linalg.qr(g.standard_normal((M, r)))
    A = np.zeros((H, M)); B = np.zeros((N, H))
    A[:r], B[:, :r] = V.T, U
    return torch.tensor(A), torch.tensor(B)


def data(M, n, seed):
    return torch.tensor(np.random.default_rng([seed, 1]).standard_normal((n, M)))


def hessian_eigs(A0, B0, S):
    N, H = B0.shape
    W0 = B0 @ A0

    def f(w):
        D = w[:N * H].reshape(N, H) @ w[N * H:].reshape(H, -1) - W0
        return 0.5 * ((D @ S) * D).sum()

    w = torch.cat([B0.reshape(-1), A0.reshape(-1)])
    return torch.linalg.eigvalsh(torch.autograd.functional.hessian(f, w))


class SGLD:
    """K localised SGLD chains started at w*. A0 (K,H,M), B0 (K,N,H); X (K,n,M) for minibatch SGLD (arm S) or None
    for the full-gradient population arm (arm P). run(T) advances every chain T steps and returns the block means
    (K, T//block) of the per-step loss K_n(w_t) on all rows (arm S) or K(w_t) (arm P); calls continue the same chains.
    A chain whose weights leave |w - w*| < 1e3 or become non-finite is marked dead (its blocks become NaN)."""

    def __init__(self, A0, B0, X, S, eps, gamma, nbeta, batch, seed, block=100):
        K, H, M = A0.shape
        self.K, self.block, self.batch = K, block, batch
        self.A0, self.B0, self.X, self.S = A0, B0, X, S       # S (K,M,M): covariance used for the recorded loss
        self.W0 = B0 @ A0
        self.eps = torch.as_tensor(np.asarray(eps, float)).view(K, 1, 1)
        self.gamma = torch.as_tensor(np.asarray(gamma, float)).view(K, 1, 1)
        self.nbeta = torch.as_tensor(np.asarray(nbeta, float)).view(K, 1, 1)
        self.sq = self.eps.sqrt()
        self.g = torch.Generator().manual_seed(int(seed))
        self.A, self.B = A0.clone(), B0.clone()
        self.alive = torch.ones(K, dtype=torch.bool)
        self.t = 0
        if X is not None:
            self.Y = X @ self.W0.transpose(1, 2)                    # noise-free targets (K,n,N)
            self.kidx = torch.arange(K).view(K, 1)
            self.n = X.shape[1]
            self.perm, self.pos = None, self.n

    def run(self, T):
        bl = self.block
        assert T % bl == 0
        A, B, A0, B0, W0, g = self.A, self.B, self.A0, self.B0, self.W0, self.g
        eps, gamma, nbeta, sq = self.eps, self.gamma, self.nbeta, self.sq
        blocks = torch.zeros(self.K, T // bl)
        for t in range(T):
            D = B @ A - W0
            if self.X is None:
                blocks[:, t // bl] += 0.5 * (D ** 2).sum((1, 2))
                gB = D @ A.transpose(1, 2)
                gA = B.transpose(1, 2) @ D
            else:
                blocks[:, t // bl] += 0.5 * ((D @ self.S) * D).sum((1, 2))      # K_n on all n rows
                if self.pos + self.batch > self.n:                                  # new epoch: fresh permutation
                    self.perm = torch.argsort(torch.rand(self.K, self.n, generator=g), dim=1)
                    self.pos = 0
                rows = self.perm[:, self.pos:self.pos + self.batch]
                self.pos += self.batch
                Xb, Yb = self.X[self.kidx, rows], self.Y[self.kidx, rows]   # (K,b,M), (K,b,N)
                XA = Xb @ A.transpose(1, 2)
                R = XA @ B.transpose(1, 2) - Yb
                gB = R.transpose(1, 2) @ XA / self.batch
                gA = (R @ B).transpose(1, 2) @ Xb / self.batch
            A = A - (eps / 2) * (nbeta * gA + gamma * (A - A0)) + sq * torch.randn(A.shape, generator=g)
            B = B - (eps / 2) * (nbeta * gB + gamma * (B - B0)) + sq * torch.randn(B.shape, generator=g)
            if t % 50 == 49:
                ok = (torch.isfinite(A).all(2).all(1) & torch.isfinite(B).all(2).all(1)
                      & ((A - A0).abs().amax((1, 2)) < 1e3) & ((B - B0).abs().amax((1, 2)) < 1e3))
                bad = self.alive & ~ok
                if bad.any():
                    self.alive &= ~bad
                    A[bad] = A0[bad]; B[bad] = B0[bad]
        self.A, self.B = A, B
        self.t += T
        blocks /= bl
        blocks[~self.alive] = float("nan")
        return blocks.numpy()

    def dist2(self):
        return (((self.A - self.A0) ** 2).sum((1, 2)) + ((self.B - self.B0) ** 2).sum((1, 2))).numpy()


def gibbs(A0, B0, S, gamma, nbeta, sweeps, seed, block=10):
    """Exact block Gibbs on exp(-nbeta L(w) - gamma/2 ||w - w*||^2), L = (1/2) tr((BA-W0) S (BA-W0)^T).
    A0 (K,H,M), B0 (K,N,H), S (K,M,M). Returns block means (K, sweeps//block) of nbeta-free L."""
    K, H, M = A0.shape
    N = B0.shape[1]
    W0 = B0 @ A0
    ds, Qs = torch.linalg.eigh(S)                                   # (K,M), (K,M,M)
    gm = torch.as_tensor(np.asarray(gamma, float)).view(K, 1, 1)
    nb = torch.as_tensor(np.asarray(nbeta, float)).view(K, 1, 1)
    g = torch.Generator().manual_seed(int(seed))
    A, B = A0.clone(), B0.clone()
    I = torch.eye(H).expand(K, H, H)
    out = torch.zeros(K, sweeps // block)
    W0S = W0 @ S
    for t in range(sweeps):
        P = nb * (A @ S @ A.transpose(1, 2)) + gm * I                # B | A: rows iid N(mean_i, P^-1)
        Lc = torch.linalg.cholesky(P)
        rhs = nb * (W0S @ A.transpose(1, 2)) + gm * B0
        mean = torch.cholesky_solve(rhs.transpose(1, 2), Lc).transpose(1, 2)
        Z = torch.randn(K, N, H, generator=g)
        B = mean + torch.linalg.solve_triangular(Lc.transpose(1, 2), Z.transpose(1, 2), upper=True).transpose(1, 2)
        dg, Qg = torch.linalg.eigh(B.transpose(1, 2) @ B)           # A | B: precision nb (S kron B^T B) + gm
        C = nb * (B.transpose(1, 2) @ W0S) + gm * A0
        lam = nb * dg.unsqueeze(2) * ds.unsqueeze(1) + gm
        At = (Qg.transpose(1, 2) @ C @ Qs) / lam + torch.randn(K, H, M, generator=g) / lam.sqrt()
        A = Qg @ At @ Qs.transpose(1, 2)
        D = B @ A - W0
        out[:, t // block] += 0.5 * ((D @ S) * D).sum((1, 2))
    return (out / block).numpy()
