"""Second estimator: Jacobian rank and Hessian (Gauss-Newton) spectrum. How they are computed and how much the
numbers in the post depend on the rank tolerance.

run_llc.py computes, at each polished checkpoint, the singular values s of the Jacobian J of the centred training
outputs (qnet.jacobian, analytic, float64). h = s^2 / N are the eigenvalues of J^T J / N, the Hessian of L_n at a
zero of the loss. Small J (min side <= 6000): numpy.linalg.svd (LAPACK gesdd), rank = #{s > 1e-9 s_max}.
Big J: eigvalsh of the smaller Gram matrix, s = sqrt(eig), rank = #{s > 1e-6 s_max}. Spectra: data/llc_traces*/.

A. Tolerance sweep on the saved spectra of the 15 verdict networks: rank, rank/2 against K(3p-2)/2, the share of
   counted directions flatter than the resolution gamma/(n beta) = 1e-4, the gap s_r / s_{r+1} at the cut.
B. Gram versus exact SVD on two Gram networks (p=23 K=92, p=31 K=93): recompute J at the same polished point,
   take exact singular values (QR, then SVD of R) and compare.
Writes data/estimator_check.csv (A) and data/estimator_check.json (A summary + B).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import qnet
from run_llc import polish, svals

HERE = Path(__file__).parent
D = HERE / "data"
RES = 1e-4  # gamma / (n beta) at the frozen setting
RTOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
VERDICT = [  # (p, K, seeds, ckpt dir, traces dir, csv)
    (11, 33, (0, 1, 2), "ckpt", "llc_traces", "llc.csv"),
    (17, 51, (0, 2), "ckpt", "llc_traces", "llc.csv"),
    (17, 68, (0, 1, 2), "ckpt", "llc_traces", "llc.csv"),
    (23, 69, (1,), "ckpt", "llc_traces", "llc.csv"),
    (23, 92, (0, 1, 2), "ckpt", "llc_traces", "llc.csv"),
    (31, 93, (0, 1, 2), "ckpt_p31_frac07", "llc_traces_p31_frac07", "llc_p31_frac07.csv"),
]


def used_rtol(method):
    return 1e-9 if method == "svd" else 1e-6


def part_a():
    rows = []
    for p, K, seeds, _, trd, csvf in VERDICT:
        meta = {(int(r["seed"])): r for r in csv.DictReader(open(D / csvf))
                if r["phase"] == "grok" and int(r["p"]) == p and int(r["K"]) == K}
        for sd in seeds:
            m = meta[sd]
            N, method = int(m["N"]), m["rank_method"]
            s = np.load(D / trd / f"p{p}_K{K}_s{sd}_grok.npz")["singular_values"]
            h = s ** 2 / N
            shift = K * (3 * p - 2)
            for rt in RTOLS:
                r = int(np.sum(s > rt * s[0]))
                gap = float(s[r - 1] / s[r]) if r < len(s) and s[r] > 0 else float("inf")
                rows.append(dict(p=p, K=K, seed=sd, N=N, method=method, rtol=rt, used=int(rt == used_rtol(method)),
                                 rank=r, rank_over_shift=r / shift, flat_share=float(np.mean(h[:r] < RES)),
                                 n_flat=int(np.sum(h[:r] < RES)), gap=gap,
                                 below_float_floor=int(method == "gram" and rt < 1e-7)))
            assert r_used(rows, p, K, sd) == int(m["rankJ"]), (p, K, sd)
    with open(D / "estimator_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def r_used(rows, p, K, sd):
    return next(r["rank"] for r in rows if (r["p"], r["K"], r["seed"]) == (p, K, sd) and r["used"])


def part_b(p, K, sd, ck, trd):
    z = np.load(D / ck / f"p{p}_K{K}_s{sd}_grok.npz")
    Xall, Yall = qnet.full_table(p)
    X, Y = Xall[z["train_idx"]], Yall[z["train_idx"]]
    W, V, _, _ = polish(z["W"], z["V"], X, Y)
    J = qnet.jacobian(W, V, X)
    t0 = time.time()
    s_gram = svals(J)  # what run_llc.py did (Gram route for this size)
    t1 = time.time()
    R = np.linalg.qr(J, mode="r")
    del J
    s_svd = np.linalg.svd(R, compute_uv=False)
    t2 = time.time()
    saved = np.load(D / trd / f"p{p}_K{K}_s{sd}_grok.npz")["singular_values"]
    rel = np.abs(s_gram - s_svd) / s_svd[0]
    out = dict(net=f"p{p}_K{K}_s{sd}", shape=[int(X.shape[0] * p), int(W.size + V.size)],
               saved_vs_recomputed_gram_max_rel=float(np.max(np.abs(saved - s_gram)) / s_gram[0]),
               gram_vs_svd_abs_err_over_smax_max=float(rel.max()),
               rank_gram_1e6=int(np.sum(s_gram > 1e-6 * s_gram[0])),
               rank_svd_1e6=int(np.sum(s_svd > 1e-6 * s_svd[0])),
               rank_svd_1e9=int(np.sum(s_svd > 1e-9 * s_svd[0])),
               shift_count=K * (3 * p - 2), cap=p * (X.shape[0] - 1),
               smallest_svd_over_smax=float(s_svd[-1] / s_svd[0]),
               flat_share_svd_1e9=float(np.mean((s_svd[: int(np.sum(s_svd > 1e-9 * s_svd[0]))] ** 2 / X.shape[0]) < RES)),
               secs_gram=round(t1 - t0, 1), secs_qr_svd=round(t2 - t1, 1))
    r9 = int(np.sum(s_svd > 1e-9 * s_svd[0]))
    out["svd_gap_at_1e9"] = float(s_svd[r9 - 1] / s_svd[r9]) if r9 < len(s_svd) else float("inf")
    out["svd_first_dropped_over_smax_1e9"] = float(s_svd[r9] / s_svd[0]) if r9 < len(s_svd) else 0.0
    out["svd_last_kept_over_smax_1e9"] = float(s_svd[r9 - 1] / s_svd[0])
    for rt in (1e-5, 1e-6, 1e-7, 1e-8):
        out[f"rank_svd_{rt:g}"] = int(np.sum(s_svd > rt * s_svd[0]))
        out[f"rank_gram_{rt:g}"] = int(np.sum(s_gram > rt * s_gram[0]))
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    rows = part_a()
    summ = {}
    for p, K, seeds, *_ in VERDICT:
        for rt in RTOLS:
            sel = [r for r in rows if r["p"] == p and r["K"] == K and r["rtol"] == rt]
            summ[f"p{p}_K{K}_rtol{rt:g}"] = dict(rank=[r["rank"] for r in sel],
                                                 flat_share=[round(r["flat_share"], 4) for r in sel],
                                                 gap=[r["gap"] for r in sel])
    res = dict(A=summ)
    if "--no-b" not in sys.argv:
        res["B"] = [part_b(23, 92, 0, "ckpt", "llc_traces"),
                    part_b(31, 93, 0, "ckpt_p31_frac07", "llc_traces_p31_frac07")]
    (D / "estimator_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("B"), indent=1))
