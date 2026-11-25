"""Secondary runs of PREREG 2(7) (reported; no effect on the status).

    .venv/bin/python extras.py s2     # S2: Lau et al. 2308.12108 App. J.1 '1k' DLN recipe on (32,32,16) r8, r12, arm P,
                                      #     3 seeds, its own 10,000 steps, last 10% kept -> data/s2_lau.json
    .venv/bin/python extras.py s1     # S1: iso-X line on the TRAINING shapes, arm S, 2 seeds -> data/s1_isoX.json

S2 mapping (research/research.md sec. 3): nbeta_eff = 8,686 x 100/3 = 2.895e5 (input variance 100/3 multiplies K),
eps 5e-7, gamma 1. S1: X = eps*nbeta/gamma fixed at 3e-4, nbeta 1000, q in {0.01, 0.1, 1, 10, 100}
(gamma = sqrt(nbeta*k4/q), eps = X*gamma/nbeta), T = max(1e4, 0.3/eps) rounded up to 1e4, burn-in T/2, 4 chains.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

import grid
from core import SGLD
from run import DATA, case_setup


def s2():
    t0 = time.time()
    nb, gm, eps, T, keep = 8686.0 * 100 / 3, 1.0, 5e-7, 10000, 1000
    rows = []
    for name in ("32x32x16r8", "32x32x16r12"):
        for sd in (0, 1, 2):
            cs = case_setup(name, sd, "P")
            K = grid.CHAINS
            ch = SGLD(cs["A"].expand(K, *cs["A"].shape).clone(), cs["B"].expand(K, *cs["B"].shape).clone(), None,
                      torch.eye(cs["M"]).expand(K, cs["M"], cs["M"]).clone(), [eps] * K, [gm] * K, [nb] * K,
                      grid.BATCH, sd, block=100)
            bl = ch.run(T)
            e_ch = nb * bl[:, (T - keep) // 100:].mean(1) / cs["lam"] - 1
            # the same chains continued to 100x the recipe length show where they are heading (diagnostic)
            bl2 = ch.run(9 * T)
            e_long = nb * bl2[:, -(5 * T) // 100:].mean(1) / cs["lam"] - 1
            rows.append({"case": name, "seed": sd, "nbeta": nb, "gamma": gm, "eps": eps, "q": nb * grid.K4 / gm ** 2,
                         "c": eps * (nb * cs["h_max"] + gm), "steps": T, "kept": keep,
                         "e_recipe": float(e_ch.mean()), "e_chains": e_ch.tolist(),
                         "e_after_1e5_steps_last_5e4": float(e_long.mean()), "lam": cs["lam"]})
            print(name, sd, f"recipe e {e_ch.mean():+.3f}; at 1e5 steps {e_long.mean():+.3f}", flush=True)
    (DATA / "s2_lau.json").write_text(json.dumps({"rows": rows, "secs": time.time() - t0, "cmd": " ".join(sys.argv)},
                                                 indent=1))


def s1():
    t0 = time.time()
    X, nb = 3e-4, 1000.0
    rows = []
    for name in grid.TRAIN:
        for sd in (0, 1):
            cs = case_setup(name, sd, "S")
            for q in (0.01, 0.1, 1, 10, 100):
                gm = math.sqrt(nb * grid.K4 / q)
                eps = X * gm / nb
                T = int(math.ceil(max(1e4, 0.3 / eps) / 1e4) * 1e4)
                K = grid.CHAINS
                ch = SGLD(cs["A"].expand(K, *cs["A"].shape).clone(), cs["B"].expand(K, *cs["B"].shape).clone(),
                          cs["X"].expand(K, *cs["X"].shape).clone(), cs["S"].expand(K, *cs["S"].shape).clone(),
                          [eps] * K, [gm] * K, [nb] * K, grid.BATCH, sd, block=100)
                bl = ch.run(T)
                e = nb * bl[:, T // 200:].mean() / cs["lam"] - 1
                rows.append({"case": name, "seed": sd, "nbeta": nb, "q": q, "gamma": gm, "eps": eps, "X": X,
                             "c": eps * (nb * cs["h_max"] + gm), "T": T, "e": float(e)})
                print(name, sd, q, f"c {rows[-1]['c']:.4f} T {T} e {e:+.3f}", flush=True)
    (DATA / "s1_isoX.json").write_text(json.dumps({"rows": rows, "secs": time.time() - t0, "cmd": " ".join(sys.argv)},
                                                  indent=1))


if __name__ == "__main__":
    torch.set_num_threads(grid.THREADS)
    {"s1": s1, "s2": s2}[sys.argv[1]]()
