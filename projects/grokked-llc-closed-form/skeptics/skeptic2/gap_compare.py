"""skeptic2: spectral gap at a planted generic teacher vs the grokked checkpoint with the same (p, K, train set).

Planted teacher: W ~ N(0,1) scaled to the checkpoint's median |w_j|, V ~ N(0,1) scaled to median |v_j|.
Reports the fraction of the rank with Gauss-Newton eigenvalue below gamma/nbeta (frozen 1e-4),
the surrogate lambda_sur at the frozen setting, and rank/2.
"""
import json
import sys

import numpy as np

from fourier_rank import jac_centred

D = "../../kran-research/projects/grokked-llc-closed-form/data/ckpt/"
NB, G = 1e4, 1.0
out = []
for name in sys.argv[1:]:
    z = np.load(D + name)
    W, V, tr = z["W"], z["V"], z["train_idx"]
    p, K = V.shape
    A, B = (g.ravel()[tr] for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
    rng = np.random.default_rng(7)
    mw, mv = np.median(np.linalg.norm(W, axis=0)), np.median(np.linalg.norm(V, axis=0))
    Wt = rng.standard_normal(W.shape); Wt *= mw / np.linalg.norm(Wt, axis=0)
    Vt = rng.standard_normal(V.shape); Vt *= mv / np.linalg.norm(Vt, axis=0)
    for tag, (WW, VV) in (("grokked", (W, V)), ("planted", (Wt, Vt))):
        h = np.linalg.svd(jac_centred(WW[:p], WW[p:], VV, A, B), compute_uv=False) ** 2 / len(tr)
        r = int((h > 1e-14 * h[0]).sum()); hr = h[:r]
        row = dict(ckpt=name, point=tag, p=p, K=K, N=len(tr), rank=r, h_max=float(hr[0]), h_min=float(hr[-1]),
                   frac_below_res=float((hr < G / NB).mean()), lambda_sur=float((NB * hr / (2 * (NB * hr + G))).sum()),
                   rank_half=r / 2, paper=K * (3 * p - 1) / 2,
                   decades=float(np.log10(hr[0] / hr[-1])))
        out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open("gap_compare.json", "w"), indent=1)
