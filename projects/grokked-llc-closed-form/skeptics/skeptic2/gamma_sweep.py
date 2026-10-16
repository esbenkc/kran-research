"""skeptic2: sensitivity of the grokked reading to the localisation gamma (nbeta, crel, steps, chains frozen).

Uses the project's own pipeline (run_llc.polish, qnet.jacobian, qnet.sgld_llc) so only gamma changes.
Also reports lambda_quad = sum 1/2 h/(h + gamma/nbeta) from the checkpoint's Gauss-Newton spectrum, and the
reading from the last quarter of each chain (a drift check: small gamma relaxes slowly).
    uv run python projects/grokked-llc-closed-form/skeptics/skeptic2/gamma_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ))
import qnet  # noqa: E402
import run_llc  # noqa: E402

CKPTS = ["p11_K33_s0", "p17_K51_s0", "p23_K92_s0"]
GAMMAS = [0.1, 1.0, 10.0, 100.0]


def one(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    name, gamma = args
    cal = json.loads((PROJ / "data" / "calibration.json").read_text())
    z = np.load(PROJ / "data" / "ckpt" / f"{name}_grok.npz")
    p = int(z["p"])
    Xall, Yall = qnet.full_table(p)
    X, Y = Xall[z["train_idx"]], Yall[z["train_idx"]]
    N = X.shape[0]
    W, V, _, _ = run_llc.polish(z["W"], z["V"], X, Y)
    s = run_llc.svals(qnet.jacobian(W, V, X))
    h = s ** 2 / N
    nb = cal["nbeta"]
    eps = cal["crel"] / (nb * h[0] + gamma)
    o = qnet.sgld_llc(W, V, X, Y, eps=eps, gamma=gamma, n_steps=cal["steps"], n_burn=cal["burn"],
                      n_chains=cal["chains"], seed=4242, nbeta=nb)
    tr = o["trace"]
    last_q = float(nb * (tr[:, -tr.shape[1] // 4:].mean() - o["L0"]))
    K = int(z["K"])
    paper = K * (3 * p - 1) / 2   # K_eff = K for these checkpoints (llc.csv)
    rJ = int(np.sum(s > (1e-9 if min(N * p, W.size + V.size) <= run_llc.GRAM_ABOVE else 1e-6) * s[0]))
    return dict(ckpt=name, p=p, K=K, N=N, gamma=gamma, nbeta=nb, eps=float(eps), lambda_hat=o["llc"],
                se=float(np.std(o["chains"], ddof=1) / 2), last_quarter=last_q, paper=paper, half_rank=rJ / 2,
                lambda_quad=float(np.sum(0.5 * h[:rJ] / (h[:rJ] + gamma / nb))), ratio=o["llc"] / paper,
                ratio_last_quarter=last_q / paper, diverged=o["diverged"])


if __name__ == "__main__":
    jobs = [(c, g) for c in CKPTS for g in GAMMAS]
    res = []
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(one, jobs):
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            res.append(r)
    (HERE / "gamma_sweep.json").write_text(json.dumps(res, indent=1))
