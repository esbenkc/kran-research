"""Numeric checks of the two displayed equations of the post (goal rule: every displayed equation gets a check).

(1) Twin-paired gap:  d'_l = mean_p (s(false_p) - s(true_p)) / sigma_l, sigma_l = pooled within-class SD of the CLEAN
    lin_l scores on the eval statements. Recomputed with plain numpy from data/scores.npz for all 28 layers x 4 states
    and compared with data/results/results.json (layers.<l>.clean / per_seed).
(2) Megaprobe score:  score = sum_{t,l} softmax_{t,l}(q_l . z_tl + c_l) (w_l . z_tl) + b. Computed in float64 numpy
    from the fitted parameters (data/fits/final_mega.pt) and the clean PCA (data/fits/pca_clean.pt) on 5 real eval
    statements read straight from the activation memmap, and compared with data/scores.npz clean__mega
    (= probes.mega_forward on MPS, fp16 storage of z).
Run: uv run python check_equations.py   -> data/results/check_equations.json; exits non-zero on a failed check.
"""
import json
import sys

import numpy as np
import torch

from common import DATA, STATES, Acts

Z = np.load(DATA / "scores.npz")
R = json.loads((DATA / "results" / "results.json").read_text())
y, split, city, src = Z["y"], Z["split"], Z["city"], Z["src"]
ev = np.where(split == "eval")[0]
out = {"dprime": {}, "mega": {}}

# (1) d'
pairs = {}
for i in ev:
    pairs.setdefault((city[i], src[i]), {})[int(y[i])] = i
pf = np.array([p[1] for p in pairs.values()])
pt = np.array([p[0] for p in pairs.values()])
worst = 0.0
for l in range(1, 29):
    sc = Z[f"clean__lin{l}"][ev]
    yc = y[ev]
    ss = sum(((sc[yc == c] - sc[yc == c].mean()) ** 2).sum() for c in (0, 1))
    sigma = np.sqrt(ss / (len(sc) - 2))
    for st in STATES:
        s = Z[f"{st}__lin{l}"]
        d = (s[pf] - s[pt]).mean() / sigma
        ref = R["layers"][str(l)]["clean"] if st == "clean" else R["layers"][str(l)]["per_seed"][int(st[-1])]
        worst = max(worst, abs(d - ref))
out["dprime"] = {"n_pairs": len(pf), "cells": 28 * 4, "max_abs_diff_vs_results_json": worst, "pass": bool(worst < 1e-9)}

# (2) megaprobe formula
P = torch.load(DATA / "fits" / "final_mega.pt")["P"]
q, c, w, b = (P[k].double().numpy() for k in ("q", "c", "w", "b"))
k = w.shape[1]
pz = torch.load(DATA / "fits" / "pca_clean.pt")
mu, V, scale = pz["mu"].double().numpy(), pz["V"].double().numpy()[:, :, :k], pz["scale"].double().numpy()[:, :k]
A = Acts("clean")
pick = ev[np.random.default_rng(0).choice(len(ev), 5, replace=False)]
rows = []
for i in pick:
    X = np.asarray(A.tok[A.off[i]:A.off[i + 1], 1:29], np.float64)  # (T, 28, d): tokens 1..n-1, layers 1..28
    z = np.einsum("tld,ldk->tlk", X - mu, V) * scale  # whitened PCA coordinates z_tl
    z = z.astype(np.float16).astype(np.float64)  # the fit stores z in fp16 (probes.pca_project)
    logits = np.einsum("tlk,lk->tl", z, q) + c
    a = np.exp(logits - logits.max())
    a /= a.sum()  # one softmax over all (token, layer) pairs
    s = float((a * np.einsum("tlk,lk->tl", z, w)).sum() + b)
    rows.append({"stmt": int(i), "numpy": s, "mega_forward": float(Z["clean__mega"][i]),
                 "abs_diff": abs(s - float(Z["clean__mega"][i]))})
md = max(r["abs_diff"] for r in rows)
out["mega"] = {"statements": rows, "max_abs_diff": md, "tol": 1e-5, "pass": bool(md < 1e-5)}
(DATA / "results" / "check_equations.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"dprime": out["dprime"], "mega_max_abs_diff": md, "mega_pass": out["mega"]["pass"]}, indent=1))
sys.exit(0 if out["dprime"]["pass"] and out["mega"]["pass"] else 1)
