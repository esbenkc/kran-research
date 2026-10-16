"""skeptic2: gamma sensitivity with MY OWN code (own polish, own Jacobian, own batched SGLD in sgld2.py).

Independent of qnet.py / run_llc.py. Frozen nbeta = 1e4, crel = 0.2, 10k steps, burn 2k, 4 chains; only gamma varies.
    cd projects/grokked-llc-closed-form/skeptics/skeptic2 && uv run python gamma_own.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fourier_rank import jac_centred  # noqa: E402
from rerun_ckpt import polish  # noqa: E402
from sgld2 import sgld_llc_qnet  # noqa: E402

D = HERE.parent.parent / "data" / "ckpt"
NB, CREL, STEPS, BURN, CH = 1e4, 0.2, 10000, 2000, 4
JOBS = [(c, g) for c in ("p11_K33_s0", "p17_K51_s0") for g in (0.1, 1.0, 10.0)]


def one(job):
    name, g = job
    z = np.load(D / f"{name}_grok.npz")
    W, V, tr = z["W"], z["V"], z["train_idx"]
    p, K = V.shape
    A, B = (x.ravel() for x in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
    Y = np.eye(p)[(A + B) % p]
    W, V, Lp, _ = polish(W, V, A[tr], B[tr], Y[tr])
    h = np.linalg.svd(jac_centred(W[:p], W[p:], V, A[tr], B[tr]), compute_uv=False) ** 2 / len(tr)
    eps = CREL / (NB * h[0] + g)
    s = sgld_llc_qnet(W, V, A[tr], B[tr], Y[tr], eps=eps, gamma=g, nbeta=NB, steps=STEPS, burn=BURN,
                      chains=CH, seed=99)
    paper = K * (3 * p - 1) / 2
    return dict(ckpt=name, gamma=g, eps=float(eps), lambda_hat=s["lambda_hat"], se=s["se"], paper=paper,
                ratio=s["lambda_hat"] / paper, polished_loss=float(Lp), diverged=s["diverged"])


if __name__ == "__main__":
    out = []
    with ProcessPoolExecutor(3) as ex:
        for r in ex.map(one, JOBS):
            print(json.dumps(r), flush=True)
            out.append(r)
    (HERE / "gamma_own.json").write_text(json.dumps(out, indent=1))
