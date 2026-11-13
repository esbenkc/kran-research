"""Vera: own numeric check of Lemma 1 (k/d overlap) and Lemma 2 (bound on |cos(v1 W, v1 (W - M))|).

Lemma 2 bound, own implementation: brute-force grid over (alpha, c) in [0,1]^2 of c*alpha subject to
c*sqrt(alpha^2 s1^2 + (1-alpha^2) s2^2) + sqrt(1-c^2)*sM >= s1. Compared with the measured |cos| (cue-free).
"""
import json
import os

import numpy as np
import torch

from vera_check import TT, TH, remainder, load, nc, target_path, v1, A_rows

OUT = os.path.dirname(os.path.abspath(__file__))


def bound(s1, s2, sM, n=2001):
    a = np.linspace(0, 1, n)[:, None]
    c = np.linspace(0, 1, n)[None, :]
    lhs = c * np.sqrt(a ** 2 * s1 ** 2 + (1 - a ** 2) * s2 ** 2) + np.sqrt(1 - c ** 2) * sM
    ok = lhs >= s1 - 1e-12
    val = np.where(ok, c * a, np.inf)
    return float(val.min())


res = {}
# Lemma 1: own Monte Carlo with Gaussian (not kaiming-uniform) entries, plus measured A0 overlap seeds 0-9
rng = np.random.default_rng(5)
mc = []
for _ in range(200):
    P = [np.linalg.qr(rng.standard_normal((1024, 48)))[0] for _ in range(2)]
    mc.append(np.trace(P[0] @ P[0].T @ P[1] @ P[1].T) / 48)
Q = []
for j in range(10):
    A = torch.cat(A_rows(nc(j, 0))).numpy()
    Q.append(np.linalg.qr(A.T)[0])
meas = [np.trace(Q[i] @ Q[i].T @ Q[j] @ Q[j].T) / 48 for i in range(10) for j in range(i + 1, 10)]
res["lemma1"] = {"pred": 48 / 1024, "mc_gauss_mean": float(np.mean(mc)), "measured_mean": float(np.mean(meas)),
                 "measured_range": [float(min(meas)), float(max(meas))]}
print(res["lemma1"])
for name, tl in (("t_h", TH), ("T", TT), ("256", [256] * 6)):
    for i in range(6):
        t = tl[i]
        W = remainder(load(target_path(i, t)), [])
        refs = [nc(j, t) for j in range(6) if j != i]
        M = sum(remainder(r, []) for r in refs) / len(refs)
        s = torch.linalg.svdvals(W)
        sM = float(torch.linalg.svdvals(M)[0])
        b = bound(float(s[0]), float(s[1]), sM)
        meas_cos = abs(float(v1(W)[0] @ v1(W - M)[0]))
        res[f"{i}|{name}"] = {"s1": float(s[0]), "s2": float(s[1]), "sM": sM, "bound": b, "measured": meas_cos,
                              "holds": meas_cos >= b - 1e-6}
        print(f"m{i} {name:3s} t={t:>3} s1 {s[0]:.4f} s2 {s[1]:.4f} sM {sM:.4f} bound {b:.3f} measured {meas_cos:.3f}")
json.dump(res, open(f"{OUT}/vera_lemma.json", "w"), indent=1)
