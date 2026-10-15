"""skeptic's own model + loss + SGLD (written from the stated claim, not from build's code).

Model (paper arXiv 2603.01192): x = [e_a; e_b] in R^{2p}, hidden h_j = (w_j . x)^2,
output f(x) = V h, W in R^{2p x K}, V in R^{p x K}. Target: one-hot of (a+b) mod p.
Loss convention (plan.md): L_n = (1/N) sum_i 1/2 ||centred residual_i||^2, no L2 term.
Which centring is meant is identified empirically in identify_loss.py.
"""
import numpy as np


def pairs(p, idx):
    idx = np.asarray(idx)
    return idx // p, idx % p


def forward(W, V, a, b, p):
    pre = W[a, :] + W[p + b, :]          # N x K, pre-activation u_a + s_b
    H = pre ** 2                          # N x K
    return H @ V.T, pre, H                # N x p


def targets(a, b, p):
    Y = np.zeros((len(a), p))
    Y[np.arange(len(a)), (a + b) % p] = 1.0
    return Y


def residual(W, V, a, b, p, Y, mode):
    F, _, _ = forward(W, V, a, b, p)
    R = F - Y
    if mode == "samples":      # centre each output over the training points
        R = R - R.mean(0, keepdims=True)
    elif mode == "outputs":    # centre each example over the p outputs
        R = R - R.mean(1, keepdims=True)
    elif mode == "none":
        pass
    return R


def loss(W, V, a, b, p, Y, mode="samples"):
    R = residual(W, V, a, b, p, Y, mode)
    return 0.5 * np.mean(np.sum(R ** 2, 1))


def grad(W, V, a, b, p, Y, freeze_W=False):
    """Gradient of the samples-centred loss. Centring is a projection, so
    dL/dF = R_c / N (projection is symmetric and idempotent)."""
    N = len(a)
    F, pre, H = forward(W, V, a, b, p)
    R = F - Y
    R = R - R.mean(0, keepdims=True)
    G = R / N                              # N x p, dL/dF
    gV = G.T @ H                           # p x K
    if freeze_W:
        return None, gV
    gH = G @ V                             # N x K
    gpre = gH * 2 * pre                    # N x K
    gW = np.zeros_like(W)
    np.add.at(gW, a, gpre)
    np.add.at(gW, p + b, gpre)
    return gW, gV


def jacobian(W, V, a, b, p, freeze_W=False):
    """Jacobian of centred outputs (N*p) w.r.t. params [W.ravel(), V.ravel()]."""
    N, K = len(a), W.shape[1]
    F, pre, H = forward(W, V, a, b, p)
    C = np.eye(N) - 1.0 / N               # centring over samples
    # dF[i,k]/dV[k',j] = delta_kk' H[i,j]
    JV = np.zeros((N, p, p, K))
    for k in range(p):
        JV[:, k, k, :] = H
    JV = np.einsum('mi,ikqj->mkqj', C, JV).reshape(N * p, p * K)
    if freeze_W:
        return JV
    # dF[i,k]/dW[r,j] = V[k,j] * 2 pre[i,j] * (r==a_i or r==p+b_i)
    JW = np.zeros((N, p, 2 * p, K))
    for i in range(N):
        g = 2 * pre[i][None, :] * V        # p x K
        JW[i, :, a[i], :] += g
        JW[i, :, p + b[i], :] += g
    JW = np.einsum('mi,ikrj->mkrj', C, JW).reshape(N * p, 2 * p * K)
    return np.concatenate([JW, JV], 1)


def hmax_power(W, V, a, b, p, freeze_W=False, iters=200, seed=0):
    """Top eigenvalue of J^T J / N by power iteration with JVP/VJP-free finite
    Jacobian (small models, so just build J)."""
    J = jacobian(W, V, a, b, p, freeze_W)
    N = len(a)
    s = np.linalg.svd(J, compute_uv=False)
    return s[0] ** 2 / N, s, J


def sgld_llc(W0, V0, a, b, p, Y, nbeta, gamma, eps, steps, burn, rng, freeze_W=False, thin=10):
    """SGLD (full batch): th <- th - eps/2 (nbeta grad L + gamma (th - th0)) + N(0, eps).
    lambda_hat = nbeta (mean_{t>burn} L(th_t) - L(th0))."""
    W, V = W0.copy(), V0.copy()
    L0 = loss(W0, V0, a, b, p, Y)
    Ls = []
    for t in range(steps):
        gW, gV = grad(W, V, a, b, p, Y, freeze_W)
        V = V - 0.5 * eps * (nbeta * gV + gamma * (V - V0)) + np.sqrt(eps) * rng.standard_normal(V.shape)
        if not freeze_W:
            W = W - 0.5 * eps * (nbeta * gW + gamma * (W - W0)) + np.sqrt(eps) * rng.standard_normal(W.shape)
        if t >= burn:
            Ls.append(loss(W, V, a, b, p, Y))
    Ls = np.array(Ls)
    return nbeta * (Ls.mean() - L0), Ls[::thin], L0
