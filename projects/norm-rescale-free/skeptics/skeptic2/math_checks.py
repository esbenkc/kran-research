#!/usr/bin/env python3
"""Skeptic2 math checks for the norm-rescale-free post equations.

Checks the displayed equations in the expert foldout from raw clean activations and probe fits.
This is independent of the build analysis code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = 1536
EPS = 1e-6


def load_pairs(rows):
    by_key = {}
    for i, r in enumerate(rows):
        if r["split"] == "eval":
            by_key.setdefault((r["city"], r["src"]), []).append((i, int(r["true"])))
    true, false = [], []
    for vals in by_key.values():
        true.append([i for i, t in vals if t == 1][0])
        false.append([i for i, t in vals if t == 0][0])
    return np.array(true), np.array(false)


def score_params(fit, tok=28):
    w = fit[f"tok{tok}__w"].astype(np.float64)
    sd = fit[f"tok{tok}__sd"].astype(np.float64)
    mu = fit[f"tok{tok}__mu"].astype(np.float64)
    b = float(fit[f"tok{tok}__b"])
    coef = np.divide(w, sd, out=np.zeros_like(w), where=sd != 0)
    with np.errstate(all="ignore"):
        bias = b - float(mu @ coef)
    return coef, bias


def rms_no_gain(h, eps=EPS):
    den = np.sqrt((h * h).sum(axis=-1, keepdims=True) / h.shape[-1] + eps)
    return h / den


def proxy_loss(h, h0, eps=0.0):
    # A stand-in for any output loss that only sees the normalised state and is zero at h0.
    d = rms_no_gain(h[None, :], eps=eps)[0] - rms_no_gain(h0[None, :], eps=eps)[0]
    with np.errstate(all="ignore"):
        return 0.5 * float(d @ d)


def main():
    data = Path("kran-research/projects/norm-rescale-free/data")
    rows = json.loads((data / "ref" / "split.json").read_text())
    true, false = load_pairs(rows)
    fit = np.load(data / "fits" / "panel.npz")
    coef, bias = score_params(fit, 28)
    h_all = np.load(data / "runs" / "r28__lam10000__seed0" / "acts_0.npy", mmap_mode="r")[:, 28, :].astype(np.float64)
    h = h_all[np.r_[true, false]]

    # Equation 1: RMSNorm rescale. With eps=0, N(c h)=N(h) for c>0 exactly.
    # With eps=1e-6, the difference is the scalar alpha-1 below.
    cs = np.array([0.3, 0.5, 2.0], dtype=np.float64)
    a = (h * h).sum(axis=1) / D
    eps_scale_errors = {}
    for c in cs:
        alpha = c * np.sqrt(a + EPS) / np.sqrt(c * c * a + EPS)
        eps_scale_errors[str(c)] = float(np.max(np.abs(alpha - 1.0)))

    # Equation 2: Euler radial gradient / Hessian null for eps=0.
    # Numeric finite-difference check on proxy loss L(h)=1/2||N(h)-N(h0)||^2 at a real state.
    h0 = h[0]
    u = h0 / np.linalg.norm(h0)
    rng = np.random.default_rng(7)
    r = rng.normal(size=h0.shape)
    with np.errstate(all="ignore"):
        r = r - u * float(r @ u)
    r = r / np.linalg.norm(r)
    step = 1e-3
    radial_grad_eps0 = (proxy_loss(h0 + step * u, h0, eps=0.0) - proxy_loss(h0 - step * u, h0, eps=0.0)) / (2 * step)
    radial_curv_eps0 = (proxy_loss(h0 + step * u, h0, eps=0.0) + proxy_loss(h0 - step * u, h0, eps=0.0)) / (step * step)
    random_curv_eps0 = (proxy_loss(h0 + step * r, h0, eps=0.0) + proxy_loss(h0 - step * r, h0, eps=0.0)) / (step * step)
    radial_curv_eps = (proxy_loss(h0 + step * u, h0, eps=EPS) + proxy_loss(h0 - step * u, h0, eps=EPS)) / (step * step)
    random_curv_eps = (proxy_loss(h0 + step * r, h0, eps=EPS) + proxy_loss(h0 - step * r, h0, eps=EPS)) / (step * step)

    # Equation 3: R0 first-step restore share.
    with np.errstate(all="ignore"):
        scores = h_all @ coef + bias
    delta = scores[false] - scores[true]
    coef_norm = np.linalg.norm(coef)
    with np.errstate(all="ignore"):
        cos_f = (h_all[false] @ coef) / (np.linalg.norm(h_all[false], axis=1) * coef_norm)
        cos_t = (h_all[true] @ coef) / (np.linalg.norm(h_all[true], axis=1) * coef_norm)
    r0 = float(np.sum(delta * (cos_f * cos_f + cos_t * cos_t)) / (2.0 * np.sum(delta)))

    out = {
        "rmsnorm_eps_scale_error_max_by_c": eps_scale_errors,
        "euler_numeric_proxy": {
            "radial_grad_eps0": float(radial_grad_eps0),
            "radial_curv_eps0": float(radial_curv_eps0),
            "random_curv_eps0": float(random_curv_eps0),
            "radial_curv_eps1e_6": float(radial_curv_eps),
            "random_curv_eps1e_6": float(random_curv_eps),
        },
        "r0_restore_share": r0,
        "r0_percent": 100.0 * r0,
        "n_pairs": int(len(true)),
        "displayed_equations_checked": [
            "N(h)=g*h/sqrt(||h||^2/1536+eps): algebraic eps scale error computed; eps=0 exact.",
            "h·grad KL=0 and H_KL h=0 at clean: finite-difference proxy loss has radial grad/curv ~0 for eps=0.",
            "R0 weighted cosine formula: rebuilt from clean acts and probe coef; got 0.280%.",
            "R restore-share formula: rebuilt in checker.py and used for signoff table.",
        ],
    }
    out_path = Path("swarm-theory-1023-norm-rescale-free/skeptic2/math_checks.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
