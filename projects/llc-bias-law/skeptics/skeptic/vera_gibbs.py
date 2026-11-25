"""Exact (step-size-free) reference for the LOCALISED tempered posterior of RRR, population loss:
p(B, A) ~ exp(-nbeta * 0.5||B A - W0||_F^2 - gamma/2 (||B - B0||^2 + ||A - A0||^2)).
B | A and A | B are Gaussian, so block Gibbs samples it exactly. E[nbeta K] under this law is what a perfect
SGLD (eps -> 0, T -> inf) would report: it isolates the localisation bias from the step and chain-length biases.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_aoyagi import aoyagi

M = N = 32
H = 16


def truth_point(r, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, r)))
    V, _ = np.linalg.qr(rng.standard_normal((M, r)))
    s = np.linspace(1.0, 2.0, r)
    W0 = (U * s) @ V.T
    B0 = np.zeros((N, H)); A0 = np.zeros((H, M))
    B0[:, :r] = U * np.sqrt(s); A0[:r] = (V * np.sqrt(s)).T
    return W0, B0, A0


def gibbs(r, gamma, nbeta, sweeps=3000, burn=500, seed=0, chains=2):
    W0, B0, A0 = truth_point(r, seed)
    rng = np.random.default_rng(seed + 100)
    I = np.eye(H)
    vals = []
    for c in range(chains):
        B, A = B0.copy(), A0.copy()
        acc, cnt = 0.0, 0
        for t in range(sweeps):
            P = nbeta * A @ A.T + gamma * I           # precision of each row of B
            L = np.linalg.cholesky(P)
            mean = np.linalg.solve(P, (nbeta * W0 @ A.T + gamma * B0).T).T
            Z = rng.standard_normal((N, H))
            B = mean + np.linalg.solve(L.T, Z.T).T     # cov = P^-1 per row
            P = nbeta * B.T @ B + gamma * I           # precision of each column of A
            L = np.linalg.cholesky(P)
            mean = np.linalg.solve(P, nbeta * B.T @ W0 + gamma * A0)
            Z = rng.standard_normal((H, M))
            A = mean + np.linalg.solve(L.T, Z)
            if t >= burn:
                D = B @ A - W0
                acc += 0.5 * np.sum(D * D); cnt += 1
        vals.append(nbeta * acc / cnt)
    lam = float(aoyagi(M, N, H, r)[0])
    return dict(r=r, gamma=gamma, nbeta=nbeta, q=nbeta / gamma ** 2, lam_true=lam,
                lam_hat=float(np.mean(vals)), chains=[float(v) for v in vals], rel_err=float(np.mean(vals) / lam - 1))


if __name__ == "__main__":
    out = []
    t0 = time.time()
    for r in (1, 8):
        for nb, g in ((1000, 10), (100, 10), (1000, 100), (10000, 1000), (1000, 1000), (100000, 100), (10, 1)):
            o = gibbs(r, float(g), float(nb), sweeps=1500, burn=300)
            out.append(o)
            print(f"GIBBS r={r} nbeta={nb:6d} gamma={g:5d} q={o['q']:.0e} g/nb={g/nb:.3f} rel={o['rel_err']:+.3f} "
                  f"chains={[round(c, 1) for c in o['chains']]} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
