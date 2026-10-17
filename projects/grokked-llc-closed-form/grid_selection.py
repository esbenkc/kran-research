"""How much does the choice among the calibration-grid settings move the grokked headline?

The calibration grid (data/calibration/calibration_grid_main.json) crossed step constant crel in {0.2, 0.5},
gamma in {1, 10} and nbeta in {1e3, 3e3, 1e4, 3e4} (16 settings). No selection rule was written down before it
ran; we took (0.2, 1, 1e4). Here we run EVERY one of the 16 settings on the lowest-seed grokked network of each
of the 6 verdict cells (same polish, chain length, burn-in, chain count and chain seed as run_llc.py) and report
the headline reading lambda_hat / (K_eff (3p-1)/2) per setting. Writes data/grid_selection.csv.
"""
from __future__ import annotations

import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import qnet  # noqa: E402
from run_llc import polish  # noqa: E402
from sensitivity import CELLS  # noqa: E402  (lowest seed of each verdict cell)

HERE = Path(__file__).parent
D = HERE / "data"
STEPS, BURN, CHAINS = 10_000, 2_000, 4  # frozen chain length (data/calibration.json)
GRID = list(itertools.product((0.2, 0.5), (1.0, 10.0), (1e3, 3e3, 1e4, 3e4)))  # (crel, gamma, nbeta)


def run(job):
    (p, K, seed, ck, trd), (crel, gamma, nbeta) = job
    t0 = time.time()
    z = np.load(D / ck / f"p{p}_K{K}_s{seed}_grok.npz")
    Xall, Yall = qnet.full_table(p)
    X, Y = Xall[z["train_idx"]], Yall[z["train_idx"]]
    N = len(X)
    W, V, _, _ = polish(z["W"], z["V"], X, Y)
    s_max = float(np.load(D / trd / f"p{p}_K{K}_s{seed}_grok.npz")["singular_values"][0])
    eps = crel / (nbeta * s_max ** 2 / N + gamma)
    o = qnet.sgld_llc(W, V, X, Y, eps=eps, gamma=gamma, n_steps=STEPS, n_burn=BURN, n_chains=CHAINS,
                      seed=10_000 + 100 * p + K + seed, nbeta=nbeta)
    paper = K * (3 * p - 1) / 2  # K_eff = K on every grokked network (llc.csv)
    return dict(p=p, K=K, seed=seed, crel=crel, gamma=gamma, nbeta=nbeta, eps=eps, lambda_hat=o["llc"],
                lambda_paper=paper, ratio=o["llc"] / paper, diverged=int(o["diverged"]),
                secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    jobs = sorted(((c, g) for c in CELLS for g in GRID), key=lambda j: -j[0][0])  # big p first
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for r in ex.map(run, jobs):
            print(r, flush=True)
            rows.append(r)
    rows.sort(key=lambda r: (r["crel"], r["gamma"], r["nbeta"], r["p"], r["K"]))
    with open(D / "grid_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
