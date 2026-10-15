"""Calibrate the SGLD LLC estimator (Lau et al., arXiv 2308.12108) on cases with a KNOWN learning coefficient.

Known-lambda cases (realisable, zero loss, generic smooth points, so lambda = rank(J)/2 exactly):
  gauss_*  : Gaussian inputs x in R^d, d = 2p. Here Cullen et al. (arXiv 2603.01192 v3) Cor 4.2 holds:
             under-param lambda = K(d+p-1)/2, over-param lambda = p d(d+1)/4.
  onehot_* : one-hot mod-add inputs, planted random teacher. Truth = rank(J)/2 (= K(3p-2)/2 below the cap p(N-1)/2).
  vonly_*  : W frozen, SGLD on V only (random-feature model of Thm 5.2). Linear in V: truth = p * rank(F~)/2.

The student starts at the teacher (L_n(w*) = 0). Step size rule (dimensionless, fixed):
  eps = crel / (nbeta * h_max + gamma),  h_max = top eigenvalue of the Hessian J^T J / n at w*.
We sweep (crel, gamma, nbeta) and weight scale, then freeze one setting for every checkpoint.

'quad' = sum_i 1/2 * h_i / (h_i + gamma/nbeta): what an exact sampler of the localised quadratic model returns.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import qnet

HERE = Path(__file__).parent
OUT = HERE / "data" / "calibration"


def hessian_spectrum(W, V, X, freeze_W=False):
    J = qnet.jacobian(W, V, X)
    if freeze_W:
        J = J[:, W.size:]
    s = np.linalg.svd(J, compute_uv=False)
    r = int(np.sum(s > 1e-6 * s[0]))
    return r, (s[:r] ** 2) / X.shape[0]


def make_case(name, p, K, n, kind, scale_w, scale_v, seed=0):
    rng = np.random.default_rng(seed)
    d = 2 * p
    if kind == "gauss":
        X = rng.standard_normal((n, d))
    else:
        X, _, _, _ = qnet.modadd_data(p, n / (p * p), seed)
    W = scale_w * rng.standard_normal((d, K))
    V = scale_v * rng.standard_normal((p, K))
    Y, _ = qnet.forward(W, V, X)          # realisable target -> zero loss at (W, V)
    freeze = kind == "vonly"
    r, h = hessian_spectrum(W, V, X, freeze_W=freeze)
    if kind == "gauss":
        D = d * (d + 1) // 2
        paper = K * (d + p - 1) / 2 if K * (d + p - 1) < p * D else p * d * (d + 1) / 4
    elif kind == "onehot":
        paper = K * (3 * p - 1) / 2
    else:
        paper = p * min(X.shape[0] - 1, K) / 2   # Thm 5.2 with l = N-1 (centred one-hot features)
    return dict(name=name, p=p, K=K, n=X.shape[0], kind=kind, W=W, V=V, X=X, Y=Y, freeze=freeze,
                rank=r, truth=r / 2, paper=paper, h=h)


CASES = [
    # name, p, K, n, kind
    ("gauss_under_p5_K3", 5, 3, 300, "gauss"),
    ("gauss_under_p7_K6", 7, 6, 400, "gauss"),
    ("gauss_over_p3_K20", 3, 20, 200, "gauss"),
    ("onehot_p11_K6", 11, 6, 60, "onehot"),
    ("onehot_p17_K12", 17, 12, 144, "onehot"),
    ("onehot_p7_K14_cap", 7, 14, 24, "onehot"),
    ("vonly_p11_K44", 11, 44, 60, "vonly"),
    ("vonly_p17_K68", 17, 68, 144, "vonly"),
]


def quad_llc(h, gamma, nbeta):
    return float(np.sum(0.5 * h / (h + gamma / nbeta)))


def step_size(crel, nbeta, h_max, gamma):
    """Fixed dimensionless rule: eps * (nbeta * h_max + gamma) = crel (stiffest direction stays stable)."""
    return crel / (nbeta * h_max + gamma)


def run_one(args):
    case_spec, scale_w, scale_v, crel, gamma, nbeta, steps, burn, chains, seed = args
    c = make_case(*case_spec, scale_w=scale_w, scale_v=scale_v)
    eps = step_size(crel, nbeta, c["h"].max(), gamma)
    t0 = time.time()
    o = qnet.sgld_llc(c["W"], c["V"], c["X"], c["Y"], eps=eps, gamma=gamma, n_steps=steps, n_burn=burn,
                      n_chains=chains, seed=seed, nbeta=nbeta, freeze_W=c["freeze"])
    lam, div = o["llc"], o["diverged"]
    return dict(case=c["name"], kind=c["kind"], p=c["p"], K=c["K"], n=c["n"], truth=c["truth"], paper=c["paper"],
                quad=quad_llc(c["h"], gamma, nbeta), lambda_hat=lam, chains=[float(x) for x in o["chains"]],
                diverged=div, rel_err=(lam - c["truth"]) / c["truth"] if not div else float("nan"),
                crel=crel, eps=float(eps), gamma=gamma, nbeta=nbeta, steps=steps, burn=burn,
                scale_w=scale_w, scale_v=scale_v, h_min=float(c["h"].min()), h_max=float(c["h"].max()),
                secs=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--crel", type=float, nargs="+", default=[0.2, 0.5])
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--nbeta", type=float, nargs="+", default=[1000.0, 3000.0, 10000.0])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", type=str, default="grid")
    ap.add_argument("--info", action="store_true", help="only print case ranks / truths / Hessian spectra")
    a = ap.parse_args()
    if a.info:
        for spec in CASES:
            c = make_case(*spec, scale_w=a.scale[0], scale_v=a.scale[0])
            print(c["name"], "n", c["n"], "rank", c["rank"], "truth", c["truth"], "paper", c["paper"],
                  "h_min %.3g h_max %.3g" % (c["h"].min(), c["h"].max()))
        raise SystemExit
    jobs = [(spec, s, s, cr, g, nb, a.steps, a.burn, a.chains, a.seed)
            for s, spec, cr, g, nb in itertools.product(a.scale, CASES, a.crel, a.gamma, a.nbeta)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
            print(r["case"], r["scale_w"], r["crel"], r["gamma"], r["nbeta"], "truth", r["truth"],
                  "hat %.2f rel %.3f" % (r["lambda_hat"], r["rel_err"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"calibration_{a.tag}.json").write_text(json.dumps(rows, indent=1))
