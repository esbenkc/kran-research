"""Stationarity check of the frozen SGLD setting at every grokked (p, K) cell.

For the lowest-seed checkpoint of each grokked cell in data/llc.csv: polish exactly as run_llc.py, then run
2 chains x 40,000 steps (4x the frozen length) with the frozen nbeta, gamma, crel. The reading is split into five
equal time blocks. A cell counts as 'settled' if the last-block reading is within 5% of the frozen 10k-step
lambda_hat of the same checkpoint (rule fixed in the swarm before this was run, plan post DECISIONS 5).

    uv run python stationarity.py            -> data/stationarity.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qnet  # noqa: E402
import run_llc  # noqa: E402

STEPS, CHAINS, TOL = 40_000, 2, 0.05


def check(args):
    os.environ["OMP_NUM_THREADS"] = "2"
    p, K, seed, lam10k = args
    meta = json.loads((HERE / "data" / "ckpt" / f"p{p}_K{K}_s{seed}.json").read_text())
    z = np.load(HERE / "data" / "ckpt" / f"p{p}_K{K}_s{seed}_grok.npz")
    Xtr, Ytr, _, _ = qnet.modadd_data(p, meta["frac"], seed)
    W, V, L, _ = run_llc.polish(z["W"], z["V"], Xtr, Ytr)
    cal = json.loads((HERE / "data" / "calibration.json").read_text())
    s = run_llc.svals(qnet.jacobian(W, V, Xtr))
    h_max = s[0] ** 2 / Xtr.shape[0]
    eps = cal["crel"] / (cal["nbeta"] * h_max + cal["gamma"])
    o = qnet.sgld_llc(W, V, Xtr, Ytr, eps=eps, gamma=cal["gamma"], n_steps=STEPS, n_burn=STEPS // 5,
                      n_chains=CHAINS, seed=777, nbeta=cal["nbeta"])
    tr = o["trace"]
    q = tr.shape[1] // 5
    blocks = [float(cal["nbeta"] * (tr[:, q * i:q * (i + 1)].mean() - o["L0"])) for i in range(5)]
    last = blocks[-1]
    return dict(p=p, K=K, seed=seed, frac=meta["frac"], lambda_hat_10k=lam10k, lambda_hat_40k=o["llc"],
                block1=blocks[0], block2=blocks[1], block3=blocks[2], block4=blocks[3], block5=last,
                rel_last_vs_10k=(last - lam10k) / lam10k, settled=int(abs(last - lam10k) / lam10k <= TOL))


if __name__ == "__main__":
    rows = list(csv.DictReader(open(HERE / "data" / "llc.csv")))
    cells = {}
    for r in rows:
        if r["phase"] != "grok" or r["lambda_hat"] in ("", "nan"):
            continue
        key = (int(r["p"]), int(r["K"]))
        if key not in cells or int(r["seed"]) < cells[key][2]:
            cells[key] = (key[0], key[1], int(r["seed"]), float(r["lambda_hat"]))
    jobs = sorted(cells.values(), key=lambda c: c[0] * c[1])
    out = []
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "5"))) as ex:
        for r in ex.map(check, jobs):
            out.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with open(HERE / "data" / "stationarity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: (r["p"], r["K"])))
