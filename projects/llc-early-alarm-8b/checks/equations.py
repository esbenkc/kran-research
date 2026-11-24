"""Checks every displayed equation of the post llc-early-alarm-8b (Selma, write). Run from the project dir:

    .venv/bin/python checks/equations.py

(1) Aoyagi-Watanabe RRR learning coefficient: sympy form of the displayed formula, its values 236 (rank 1) and 312
    (rank 8) at M = N = 32, H = 16, and agreement with gate.aoyagi on every case-1 shape M, N, H, r <= 12.
(2) n beta = n / ln n = 26.38 at n = 128.
(3) Step rule eps = c / (n beta h_max + gamma): recomputed for every row of data/gate/grid.json.
(4) Estimator lambda_hat = n beta (mean over the C x D post-burn-in minibatch losses - L_n(w*)): recomputed from the
    saved per-step losses of every row of data/gate/grid.json.
(5) SGLD update: one step of sgld.sgld_chain on a toy quadratic equals the displayed update with the same noise draw.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from gate import aoyagi  # noqa: E402
from sgld import sgld_chain  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# (1) displayed formula: lambda = (2(H+r)(M+N) - (M-N)^2 - (H+r)^2 + delta) / 8, delta = 1 if M+N+H+r odd
M, N, H, r, delta = sp.symbols("M N H r delta", integer=True)
lam_expr = (2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2 + delta) / 8


def lam_post(m, n, h, rr):
    d = 1 if (m + n + h + rr) % 2 else 0
    return lam_expr.subs({M: m, N: n, H: h, r: rr, delta: d})


check("Aoyagi rank 1 = 236", lam_post(32, 32, 16, 1) == 236, str(lam_post(32, 32, 16, 1)))
check("Aoyagi rank 8 = 312", lam_post(32, 32, 16, 8) == 312, str(lam_post(32, 32, 16, 8)))
n_shapes, mism = 0, 0
for m in range(1, 13):
    for n in range(1, 13):
        for h in range(1, 13):
            for rr in range(0, h + 1):
                if n + rr <= m + h and m + rr <= n + h and h + rr <= m + n:  # case 1 of Aoyagi-Watanabe
                    n_shapes += 1
                    mism += sp.nsimplify(lam_post(m, n, h, rr)) != sp.nsimplify(aoyagi(m, n, h, rr)[0])
check("displayed formula == gate.aoyagi on case-1 shapes", mism == 0, f"{n_shapes} shapes, {mism} mismatches")

# (2) n beta
nb = 128 / math.log(128)
check("n beta = 128 / ln 128 = 26.38", round(nb, 2) == 26.38, f"{nb:.4f}")

# (3) + (4) from the saved gate grid
g = json.loads((HERE / "data/gate/grid.json").read_text())
rows = g["rows"]
eps_err, lam_err = 0.0, 0.0
for x in rows:
    eps = x["c"] / (x["nbeta"] * x["h_max"] + x["gamma"])
    eps_err = max(eps_err, abs(eps - x["eps"]) / x["eps"])
    draws = [l[25:] for l in x["losses"]]
    lam = x["nbeta"] * (float(np.mean(np.concatenate(draws))) - x["L_wstar"])
    if math.isfinite(x["lambda_hat"]):
        lam_err = max(lam_err, abs(lam - x["lambda_hat"]) / max(1.0, abs(x["lambda_hat"])))
check("step rule eps = c / (n beta h_max + gamma) on all grid rows", eps_err < 1e-12, f"{len(rows)} rows, max rel err {eps_err:.1e}")
check("lambda_hat = n beta (mean draws - L_n(w*)) on all finite grid rows", lam_err < 1e-9, f"max rel err {lam_err:.1e}")

# (5) one SGLD step on L(w) = 0.5 * mean_rows (w - a_row)^2, compared with the displayed update and the same noise
torch.manual_seed(0)
a = torch.randn(16, 3, dtype=torch.float64)
w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
w_star = [torch.full((3,), 0.3, dtype=torch.float64)]
nbeta, eps, gamma, batch, seed = 5.0, 0.01, 2.0, 4, 7


def loss_rows(rows):
    return 0.5 * ((w - a[rows]) ** 2).sum(1).mean()


out = sgld_chain(loss_rows, [w], w_star, 16, batch, nbeta, eps, gamma, burn=0, draws=1, seed=seed)
# replay the generator: permutation first, then one noise draw (sgld_chain's order)
gen = torch.Generator(device="cpu").manual_seed(seed)
perm = torch.randperm(16, generator=gen)
rows = perm[:batch].tolist()
xi = torch.randn((3,), generator=gen, dtype=torch.float32).to(torch.float64)
w0 = w_star[0].clone()
grad = (w0 - a[rows]).mean(0)
w1 = w0 - (eps / 2) * (nbeta * grad + gamma * (w0 - w_star[0])) + math.sqrt(eps) * xi
L1 = float(0.5 * ((w0 - a[rows]) ** 2).sum(1).mean())
check("SGLD loss at the start of the step matches", abs(out["losses"][0] - L1) < 1e-12, f"{out['losses'][0]:.6f} vs {L1:.6f}")


# re-run one step by hand through sgld_chain's own parameter update to compare the end point
def end_point():
    return w.detach().clone()


out2 = sgld_chain(loss_rows, [w], w_star, 16, batch, nbeta, eps, gamma, burn=0, draws=1, seed=seed, end_fn=end_point)
diff = float((out2["end"] - w1).abs().max())
check("one SGLD step == displayed update (same noise)", diff < 1e-12, f"max abs diff {diff:.1e}")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
