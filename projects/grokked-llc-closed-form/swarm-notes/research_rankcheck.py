"""Research sanity check (not the experiment): Jacobian rank of the quadratic net
f_c(a,b) = sum_j V[c,j] * (W[:,j] . [e_a; e_b])^2 at RANDOM parameters, on one-hot
modular-addition inputs, vs. the paper's identifiable-dimension counts.

At a smooth zero-loss point where the Jacobian has locally constant rank r, LLC = r/2
(Watanabe; also the paper's Thm F.1 / Remark G.3). So rank/2 is the 'generic' LLC that the
paper's Cor 4.2 / G.10 claims equals min(K(3p-1), p*d(d+1)/2)/2 with d = 2p.
"""
import itertools
import numpy as np

rng = np.random.default_rng(0)


def jac(p, K, pairs, centre=True):
    d = 2 * p
    W = rng.normal(size=(d, K))
    V = rng.normal(size=(p, K))
    X = np.zeros((len(pairs), d))
    for i, (a, b) in enumerate(pairs):
        X[i, a] = 1.0
        X[i, p + b] = 1.0
    Z = X @ W  # N x K  pre-activations
    N = len(pairs)
    # rows indexed (sample i, output c); columns: W (d*K) then V (p*K)
    J = np.zeros((N, p, d * K + p * K))
    for c in range(p):
        # d f_c / d W[:, j] = 2 V[c,j] Z[:, j] x
        dW = 2 * (V[c][None, :] * Z)[:, None, :] * X[:, :, None]  # N x d x K
        J[:, c, : d * K] = dW.reshape(N, -1)
        J[:, c, d * K + c * K : d * K + (c + 1) * K] = Z**2
    if centre:  # P_perp along the sample dimension
        J = J - J.mean(axis=0, keepdims=True)
    return np.linalg.matrix_rank(J.reshape(N * p, -1))


for p in (5, 7):
    full = list(itertools.product(range(p), range(p)))
    idx = rng.permutation(len(full))
    train = [full[i] for i in idx[: int(0.4 * len(full))]]
    for K in (1, 2, 4, 8, 16, 32, 64):
        r_full = jac(p, K, full)
        r_tr = jac(p, K, train)
        paper = min(K * (3 * p - 1), p * (2 * p) * (2 * p + 1) // 2)
        print(
            f"p={p} K={K:3d}  rank(full p^2, centred)={r_full:5d}  rank(train N={len(train)}, centred)={r_tr:5d}"
            f"  paper K(3p-1)|pd(d+1)/2 -> {paper:5d}   K(3p-2)={K*(3*p-2):5d}  p(p^2-1)={p*(p*p-1)}  p(N-1)={p*(len(train)-1)}"
        )
