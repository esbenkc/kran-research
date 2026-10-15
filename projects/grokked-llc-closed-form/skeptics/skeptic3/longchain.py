"""skeptic3 null test: is lambda_hat at a grokked checkpoint converged in chain length?

For one checkpoint, run 2 chains of `steps` with my frozen rule (gamma=1, eps=0.1/(nbeta*h_max)) and report
nbeta*(mean L - L0) in consecutive windows of 10% of the run. A converged estimate has flat window means.
Usage: longchain.py <ckpt.npz> <nbeta> <steps> <out.json>
"""
import json
import sys

import numpy as np

from rerun import gn_lmax, load, targets
from sgld import grad, loss


def main(path, nbeta, steps, out, chains=2, gamma=1.0):
    p, U0, S0, V0, tr, te, z = load(path)
    A, B, Y = targets(p, tr)
    lmax = gn_lmax(U0, S0, V0, A, B)
    eps = 0.1 / (nbeta * lmax)
    L0 = loss(U0, S0, V0, A, B, Y)
    OA, OB = np.eye(p)[A], np.eye(p)[B]
    res = dict(file=path.split("/")[-1], nbeta=nbeta, steps=steps, eps=eps, lmax=lmax, chains=[])
    for ch in range(chains):
        rng = np.random.default_rng(100 + ch)
        U, S, V = U0.copy(), S0.copy(), V0.copy()
        Ls = np.empty(steps)
        for t in range(steps):
            gU, gS, gV, L = grad(U, S, V, A, B, Y, OA, OB)
            Ls[t] = L
            se = np.sqrt(eps)
            V = V - 0.5 * eps * (nbeta * gV + gamma * (V - V0)) + se * rng.standard_normal(V.shape)
            U = U - 0.5 * eps * (nbeta * gU + gamma * (U - U0)) + se * rng.standard_normal(U.shape)
            S = S - 0.5 * eps * (nbeta * gS + gamma * (S - S0)) + se * rng.standard_normal(S.shape)
        w = steps // 10
        windows = [float(nbeta * (Ls[i * w:(i + 1) * w].mean() - L0)) for i in range(10)]
        drift = float(np.sqrt(((U - U0) ** 2).sum() + ((S - S0) ** 2).sum() + ((V - V0) ** 2).sum())
                      / np.sqrt((U0 ** 2).sum() + (S0 ** 2).sum() + (V0 ** 2).sum()))
        res["chains"].append(dict(window_means=windows, rel_drift_end=drift))
        print(res["file"], nbeta, steps, "chain", ch, [round(x, 1) for x in windows], "drift", round(drift, 3), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
