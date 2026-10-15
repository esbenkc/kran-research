"""skeptic3: validate own SGLD on planted generic teachers where truth = rank(J)/2 (smooth point).

Also checks the analytic gradient against finite differences.
"""
import json
import sys

import numpy as np

from rank_check import jacobian, rank
from sgld import grad, llc, loss


def planted(p, K, frac, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)])
    N = int(round(frac * p * p))
    idx = rng.permutation(len(pairs))[:N]
    A, B = pairs[idx, 0], pairs[idx, 1]
    U = scale * rng.normal(size=(K, p))
    S = scale * rng.normal(size=(K, p))
    V = scale * rng.normal(size=(p, K))
    Y = V @ ((U[:, A] + S[:, B]) ** 2)
    return U, S, V, A, B, Y


def gradcheck():
    U, S, V, A, B, Y = planted(5, 3, 0.8, 1)
    Y = Y + np.random.default_rng(2).normal(size=Y.shape)
    gU, gS, gV, _ = grad(U, S, V, A, B, Y)
    h = 1e-6
    err = 0
    for M, g in ((U, gU), (S, gS), (V, gV)):
        for idx in [(0, 0), (1, 2), (2, 1)]:
            M[idx] += h
            lp = loss(U, S, V, A, B, Y)
            M[idx] -= 2 * h
            lm = loss(U, S, V, A, B, Y)
            M[idx] += h
            err = max(err, abs((lp - lm) / (2 * h) - g[idx]) / (abs(g[idx]) + 1e-9))
    return err


if __name__ == "__main__":
    print("max rel grad err", gradcheck())
    rows = []
    # Settings rule (frozen for all later use): gamma=1, eps = 0.1 / (nbeta * lmax(J^T J / N)),
    # 20000 steps, 5000 burn.  Naive eps (no Hessian scaling) diverges: eps*nbeta*lmax must be < 2.
    for p, K, frac in [(7, 2, 1.0), (7, 4, 0.5), (11, 3, 0.5), (17, 4, 0.5), (11, 8, 1.0)]:
        U, S, V, A, B, Y = planted(p, K, frac, 0)
        J = jacobian(U, S, V, A, B)
        truth = rank(J) / 2
        lmax = np.linalg.eigvalsh(J.T @ J / len(A)).max()
        for nbeta in (1e3, 1e4, 1e5):
            r = llc(U, S, V, A, B, Y, nbeta=nbeta, eps=0.1 / (nbeta * lmax), gamma=1.0, steps=20000, burn=5000,
                    chains=4, seed=0)
            row = dict(p=p, K=K, N=len(A), truth=truth, nbeta=nbeta, **r)
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "per_chain"}), flush=True)
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "validate_sgld.json", "w"), indent=1)
