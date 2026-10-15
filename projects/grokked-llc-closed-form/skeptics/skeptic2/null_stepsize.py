"""skeptic2 null tests at the frozen setting (nbeta=1e4, gamma=1, eps = crel/(nbeta h_max + gamma)).

(A) Regular null. A REGULAR quadratic loss with exactly the grokked checkpoint's Gauss-Newton spectrum
    (true LLC = rank/2 by construction). If SGLD on it reads lambda_sur, not rank/2, then the gap
    at the grokked point is a resolution effect that a perfectly regular model shows too.
(B) Step-size sweep. crel in {0.05, 0.1, 0.2, 0.4} on the real (polished) checkpoint. If lambda_hat
    moves by less than a few % the gap is not a discretisation artefact.

Usage: python null_stepsize.py p17_K51_s0_grok.npz [more ckpts]
"""
import json
import sys

import numpy as np

from fourier_rank import jac_centred
from rerun_ckpt import polish
from sgld2 import sgld_llc_qnet, sgld_llc_quadratic

D = "../../kran-research/projects/grokked-llc-closed-form/data/ckpt/"
NB, G, STEPS, CH = 1e4, 1.0, 10000, 4
out = []
for name in sys.argv[1:]:
    z = np.load(D + name)
    W, V, tr = z["W"], z["V"], z["train_idx"]
    p, K = V.shape
    A, B = (g.ravel() for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
    Y = np.eye(p)[(A + B) % p]
    W, V, Lp, _ = polish(W, V, A[tr], B[tr], Y[tr])
    h = np.linalg.svd(jac_centred(W[:p], W[p:], V, A[tr], B[tr]), compute_uv=False) ** 2 / len(tr)
    r = int((h > 1e-14 * h[0]).sum()); h = h[:r]
    for crel in (0.05, 0.1, 0.2, 0.4):
        eps = crel / (NB * h[0] + G)
        q = sgld_llc_quadratic(h, eps, G, NB, STEPS, STEPS // 5, CH, seed=1)
        s = sgld_llc_qnet(W, V, A[tr], B[tr], Y[tr], eps=eps, gamma=G, nbeta=NB, steps=STEPS, burn=STEPS // 5,
                          chains=CH, seed=2)
        row = dict(ckpt=name, crel=crel, eps=eps, rank_half=r / 2, paper=K * (3 * p - 1) / 2,
                   regular_sgld=q["lambda_hat"], regular_se=q["se"], regular_exact_discrete=q["exact_discrete"],
                   regular_exact_continuum=q["exact_continuum"],
                   qnet_sgld=s["lambda_hat"], qnet_se=s["se"], qnet_diverged=s["diverged"])
        out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open("null_stepsize.json", "w"), indent=1)
