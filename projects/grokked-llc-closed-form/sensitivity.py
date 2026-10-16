"""Sensitivity of the grokked LLC reading to gamma and n*beta (follow-up to G. Wang's sampler question).

For the lowest-seed checkpoint of each of the 6 verdict cells: polish exactly as run_llc.py, then run SGLD with the
frozen calibration setting (crel, steps, burn, chains from data/calibration.json), changing ONE of gamma or nbeta
(SETTINGS). The frozen point (gamma 1, nbeta 1e4) is lambda_hat in llc.csv with the same chain seed. The step size
follows the frozen rule eps = crel / (nbeta * h_max + gamma), h_max = s_max(J)^2 / N from data/llc_traces*/.
Writes data/sensitivity.csv.
"""
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import qnet  # noqa: E402
from run_llc import polish  # noqa: E402

HERE = Path(__file__).parent
D = HERE / "data"
SETTINGS = [(0.1, 1e4), (10.0, 1e4), (1.0, 1e3), (1.0, 1e5)]  # (gamma, nbeta), one change from (1, 1e4) each
CELLS = [  # (p, K, seed, ckpt dir, traces dir) = lowest seed of each verdict cell
    (11, 33, 0, "ckpt", "llc_traces"),
    (17, 51, 0, "ckpt", "llc_traces"),
    (17, 68, 0, "ckpt", "llc_traces"),
    (23, 69, 1, "ckpt", "llc_traces"),
    (23, 92, 0, "ckpt", "llc_traces"),
    (31, 93, 0, "ckpt_p31_frac07", "llc_traces_p31_frac07"),
]


def run(job):
    (p, K, seed, ck, trd), (gamma, nbeta) = job
    t0 = time.time()
    cal = json.loads((D / "calibration.json").read_text())
    z = np.load(D / ck / f"p{p}_K{K}_s{seed}_grok.npz")
    Xall, Yall = qnet.full_table(p)
    X, Y = Xall[z["train_idx"]], Yall[z["train_idx"]]
    N = len(X)
    W, V, _, _ = polish(z["W"], z["V"], X, Y)
    s_max = float(np.load(D / trd / f"p{p}_K{K}_s{seed}_grok.npz")["singular_values"][0])
    eps = cal["crel"] / (nbeta * s_max ** 2 / N + gamma)
    o = qnet.sgld_llc(W, V, X, Y, eps=eps, gamma=gamma, n_steps=cal["steps"], n_burn=cal["burn"],
                      n_chains=cal["chains"], seed=10_000 + 100 * p + K + seed, nbeta=nbeta)
    c = np.array(o["chains"])
    return dict(p=p, K=K, seed=seed, gamma=gamma, nbeta=nbeta, eps=eps, lambda_hat=o["llc"],
                lambda_se=float(c.std(ddof=1) / np.sqrt(len(c))), lambda_paper=K * (3 * p - 1) / 2,
                diverged=int(o["diverged"]), secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    jobs = [(c, s) for c in CELLS for s in SETTINGS]
    jobs.sort(key=lambda j: -j[0][0])  # big p first
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for r in ex.map(run, jobs):
            print(r, flush=True)
            rows.append(r)
    rows.sort(key=lambda r: (r["p"], r["K"], r["gamma"], r["nbeta"]))
    with open(D / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
