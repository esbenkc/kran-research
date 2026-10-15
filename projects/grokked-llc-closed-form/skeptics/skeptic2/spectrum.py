"""skeptic2 null test: the 'regular surrogate'.

At a checkpoint, L_n(w*+d) ~ 1/2 d^T G d with G = J^T J / N (Gauss-Newton; exact Hessian at a zero-loss point).
If the loss were exactly this quadratic (a REGULAR model of dimension rank G), the SGLD estimator with
(nbeta, gamma) would read, in the eps -> 0 limit,
    lambda_sur = sum_i nbeta h_i / (2 (nbeta h_i + gamma))      (h_i = eigenvalues of G)
and the exact discrete-time value for step eps is
    lambda_sur_eps = sum_i nbeta h_i / (2 a_i (1 - eps a_i / 4)),  a_i = nbeta h_i + gamma.
lambda_sur <= rank/2 always. Comparing SGLD at the real checkpoint to lambda_sur separates
'estimator resolution' (small h_i < gamma/nbeta are invisible) from genuine singular structure
(higher-order terms, which make SGLD differ from lambda_sur).
Also reports the number of GN eigenvalues above gamma/nbeta (the estimator's effective dimension).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from fourier_rank import jac_centred


def gn_eigs(W, V, A, B):
    p = V.shape[0]
    J = jac_centred(W[:p], W[p:], V, A, B)
    s = np.linalg.svd(J, compute_uv=False)
    return s**2 / len(A)


def surrogate(h, nbeta, gamma, eps=None):
    a = nbeta * h + gamma
    cont = float((nbeta * h / (2 * a)).sum())
    disc = None if eps is None else float((nbeta * h / (2 * a * (1 - eps * a / 4))).sum())
    return cont, disc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--glob", default="*_grok.npz")
    ap.add_argument("--nbeta", type=float, nargs="+", default=[1e3, 1e4])
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    ap.add_argument("--out", default="spectrum.jsonl")
    a = ap.parse_args()
    with open(a.out, "a") as f:
        for path in sorted(glob.glob(os.path.join(a.ckpt_dir, a.glob))):
            mt = re.search(r"p(\d+)_K(\d+)_s(\d+)_(mem|grok)", os.path.basename(path))
            p, K = int(mt[1]), int(mt[2])
            z = np.load(path)
            Aa, Bb = (g.ravel() for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
            tr = z["train_idx"]
            h = gn_eigs(z["W"], z["V"], Aa[tr], Bb[tr])
            N = len(tr)
            row = dict(file=os.path.basename(path), p=p, K=K, N=N, h_max=float(h[0]),
                       rank_1e7=int((h > 1e-14 * h[0]).sum()),
                       quantiles={q: float(np.quantile(h[h > 1e-14 * h[0]], q)) for q in (0.01, 0.1, 0.5)},
                       lambda_paper=K * (3 * p - 1) / 2, lambda_cap=p * (N - 1) / 2)
            for nb in a.nbeta:
                for g in a.gamma:
                    row[f"sur_nb{nb:g}_g{g:g}"] = surrogate(h, nb, g)[0]
                    row[f"ndim_nb{nb:g}_g{g:g}"] = int((nb * h > g).sum())
            np.save(path.replace(os.path.dirname(path), ".").replace(".npz", "_gn.npy"), h)
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")
