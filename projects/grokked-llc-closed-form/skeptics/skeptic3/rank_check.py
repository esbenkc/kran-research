"""skeptic3: independent check of the closed-form LLC counts in arXiv 2603.01192.

Model (paper Sec. 3): f_c(a,b) = sum_j V[c,j] * (w_j . x)^2,  x = [e_a; e_b] in R^{2p}.
With one-hot x, w_j . x = u_j[a] + s_j[b] where w_j = [u_j; s_j].
Loss: centred squared loss over the N training pairs (P_perp along the sample axis).

For a realisable squared loss, LLC >= rank(J)/2, with equality when the zero set is smooth
at w* (minimally singular).  J = Jacobian of the centred training outputs w.r.t. theta.
So rank(J) at a generic point is the generic "2*lambda".

Paper counts (Cor. 4.2 / 5.4): per neuron d+p-1 = 3p-1 directions; cap p*d(d+1)/2.
This script counts what the one-hot data actually sees.
"""
import json
import sys

import numpy as np


def jacobian(U, S, V, A, B):
    """Centred Jacobian of outputs (p x N) wrt (U, S, V). U,S: K x p, V: p x K."""
    K, p = U.shape
    N = len(A)
    H = U[:, A] + S[:, B]  # K x N
    # d f_c(n) / d V[c', j] = delta_cc' H[j,n]^2
    # d f_c(n) / d U[j, a]  = 2 V[c,j] H[j,n] [A_n == a]
    rows = p * N
    cols = 3 * K * p
    J = np.zeros((p, N, 3, K, p))
    onehotA = np.eye(p)[A]  # N x p
    onehotB = np.eye(p)[B]
    for c in range(p):
        G = 2 * V[c][:, None] * H  # K x N : d f_c / d h_j
        J[c, :, 0] = np.einsum("jn,na->nja", G, onehotA)
        J[c, :, 1] = np.einsum("jn,nb->njb", G, onehotB)
        # V block: only column c of V matters for output c
        J[c, :, 2, :, c] = (H ** 2).T
    J = J.reshape(p, N, cols)
    J = J - J.mean(axis=1, keepdims=True)  # centre along sample axis, per output
    return J.reshape(rows, cols)


def rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    tol = s.max() * max(M.shape) * np.finfo(float).eps * 10
    return int((s > tol).sum())


def run(p, K, frac, seed=0):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)])
    N = int(round(frac * p * p))
    idx = rng.permutation(len(pairs))[:N]
    A, B = pairs[idx, 0], pairs[idx, 1]
    U = rng.normal(size=(K, p))
    S = rng.normal(size=(K, p))
    V = rng.normal(size=(p, K))
    r = rank(jacobian(U, S, V, A, B))
    d = 2 * p
    return dict(
        p=p, K=K, N=N, frac=frac, rank=r,
        paper_generic=min(K * (3 * p - 1), p * d * (d + 1) // 2),
        mine_generic=min(K * (3 * p - 2), p * (N - 1)),
        n_params=3 * K * p,
    )


if __name__ == "__main__":
    out = []
    for p, K, frac in [(5, 1, 1.0), (7, 1, 1.0), (11, 1, 1.0), (7, 1, 0.5),
                       (7, 4, 1.0), (7, 8, 1.0), (7, 16, 1.0), (7, 40, 1.0),
                       (7, 4, 0.5), (7, 8, 0.5), (7, 16, 0.5),
                       (11, 8, 1.0), (11, 30, 1.0), (11, 60, 1.0),
                       (11, 8, 0.5), (11, 30, 0.5)]:
        res = run(p, K, frac)
        out.append(res)
        print(json.dumps(res), flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "rank_check.json", "w"), indent=1)
