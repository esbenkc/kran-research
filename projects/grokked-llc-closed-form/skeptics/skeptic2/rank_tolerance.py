"""skeptic2: is 'rank J = K(3p-2)' at grokked checkpoints robust to the SVD tolerance and to polishing?

For each ckpt: Jacobian singular values before and after build-style polishing (Adam, wd=0, until L<1e-8),
and the rank at relative tolerances 1e-4..1e-10 (relative to the top singular value).
"""
import json
import sys

import numpy as np

from fourier_rank import jac_centred
from rerun_ckpt import polish

D = "../../kran-research/projects/grokked-llc-closed-form/data/ckpt/"
out = []
for name in sys.argv[1:]:
    z = np.load(D + name)
    W, V, tr = z["W"], z["V"], z["train_idx"]
    p, K = V.shape
    A, B = (g.ravel() for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
    Y = np.eye(p)[(A + B) % p]
    W2, V2, Lp, steps = polish(W, V, A[tr], B[tr], Y[tr])
    for tag, (WW, VV) in (("ckpt", (W, V)), ("polished", (W2, V2))):
        s = np.linalg.svd(jac_centred(WW[:p], WW[p:], VV, A[tr], B[tr]), compute_uv=False)
        ranks = {f"{t:.0e}": int((s > t * s[0]).sum()) for t in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10)}
        row = dict(ckpt=name, point=tag, generic=min(K * (3 * p - 2), p * (len(tr) - 1)), ranks=ranks,
                   polish_loss=Lp, polish_steps=steps)
        out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open("rank_tolerance.json", "w"), indent=1)
