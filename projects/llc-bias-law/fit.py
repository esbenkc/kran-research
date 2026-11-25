"""Fit the frozen law L (and secondary L_z, nulls N1-N4) on the TRAINING cells only, then write predictions.json
for every held-out cell BEFORE any held-out cell runs (PREREG 2(4), 2(6), 2(7); order of work step 3).

    .venv/bin/python fit.py            # reads data/cells_S/<train>.json, data/timing_S.json -> predictions.json

Law L:   lambda_pred = sum_i (1/2) [1/(1+g_i)] / [1 - c_i/4]  +  lambda0 * psi(q),  psi(q) = 1/(1+(a/q)^b)
         g_i = gamma/(nbeta h_i), c_i = eps (nbeta h_i + gamma), h_i = positive eigenvalues of the POPULATION
         Hessian of K at w*, q = nbeta*k4/gamma^2 (k4 = 1/2). e_pred = lambda_pred/lambda - 1.
         a, b: least squares in e (points) over TRAIN cells with q >= 0.1, arm S, seed-mean e.
L_z:     same regular term + lambda0 * Phi(z), Phi(z) = z/(1+z^b)^(1/b), z = q M'N'H'/lambda0; b on TRAIN, all q.
N1: mean TRAIN e (q >= 0.1).  N2: e + 1 = 1/(1+(a/q)^b) on total lambda.  N3: isotonic e vs log X, X = eps nbeta/gamma
(direction = the one with the smaller training SSE; linear interpolation in log X, flat extrapolation).
N4: nearest TRAIN cell in (log nbeta, log gamma, log eps); ties = mean.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq, isotonic_regression, least_squares

import grid
from core import hessian_eigs, teacher

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "research"))
from rrr_lambda import rrr_lambda  # noqa: E402

DATA = HERE / "data"
BLIND = ["24x24x12r3", "24x24x12r6", "48x48x8r2", "20x20x10r5", "40x24x12r4"]
SEEN = ["32x32x16r2", "32x32x16r4", "32x32x16r12", "16x16x8r1", "16x16x8r2", "16x16x8r4"]
CONTROL = ["16x16x8r8"]
S2_SHAPES = ["32x32x16r8", "32x32x16r12"]


def parse_case(name):
    shape, r = name.split("r")
    M, N, H = (int(v) for v in shape.split("x"))
    return M, N, H, int(r)


_SHAPE = {}


def shape_info(name):
    """Deterministic properties of a shape (no sampling): truth, null block, population Hessian eigenvalues."""
    if name in _SHAPE:
        return _SHAPE[name]
    M, N, H, r = parse_case(name)
    A, B = teacher(M, N, H, r, 0)
    ev = hessian_eigs(A, B, torch.eye(M, dtype=torch.float64)).numpy()
    h = ev[ev > 1e-8 * ev.max()]
    assert len(h) == r * (M + N - r), (name, len(h))
    lam, lam0 = float(rrr_lambda(M, N, H, r)), float(rrr_lambda(M - r, N - r, H - r, 0))
    info = {"name": name, "M": M, "N": N, "H": H, "r": r, "lam": lam, "lam0": lam0, "h": h, "h_max": float(ev.max()),
            "MNH0": (M - r) * (N - r) * (H - r)}
    _SHAPE[name] = info
    return info


def reg_term(info, nbeta, gamma, eps):
    h = info["h"]
    g = gamma / (nbeta * h)
    c = eps * (nbeta * h + gamma)
    return float(np.sum(0.5 / (1 + g) / (1 - c / 4)))


def psi(q, a, b):
    return 1.0 / (1.0 + (a / q) ** b)


def Phi(z, b):
    return z / (1.0 + z ** b) ** (1.0 / b)


def cell_eps(info, nbeta, gamma, c):
    return c / (nbeta * info["h_max"] + gamma)


def e_L(info, s, a, b):
    eps = cell_eps(info, s["nbeta"], s["gamma"], s["c"])
    return (reg_term(info, s["nbeta"], s["gamma"], eps) + info["lam0"] * psi(s["q"], a, b)) / info["lam"] - 1


def e_Lz(info, s, b):
    eps = cell_eps(info, s["nbeta"], s["gamma"], s["c"])
    null = 0.0 if info["lam0"] == 0 else info["lam0"] * Phi(s["q"] * info["MNH0"] / info["lam0"], b)
    return (reg_term(info, s["nbeta"], s["gamma"], eps) + null) / info["lam"] - 1


def e_N2(s, a, b):
    return psi(s["q"], a, b) - 1


class Iso:
    """Isotonic fit of y vs log x (direction with the smaller SSE), linear interpolation, flat extrapolation."""

    def __init__(self, x, y):
        o = np.argsort(x)
        lx, yy = np.log(np.asarray(x)[o]), np.asarray(y)[o]
        best = None
        for inc in (True, False):
            fit = isotonic_regression(yy, increasing=inc).x
            sse = float(((fit - yy) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, inc, fit)
        self.sse, self.increasing, fit = best
        ux, idx = np.unique(lx, return_inverse=True)
        self.lx = ux
        self.fy = np.array([fit[idx == k].mean() for k in range(len(ux))])

    def __call__(self, x):
        return float(np.interp(math.log(x), self.lx, self.fy))


def settings():
    out = []
    for nb in grid.NBETA:
        for q in grid.Q:
            for c in grid.C:
                out.append({"nbeta": nb, "q": q, "c": c, "gamma": math.sqrt(nb * grid.K4 / q)})
    return out


def skey(s):
    return f"nb{s['nbeta']:g}_q{s['q']:g}_c{s['c']:g}"


def train_table():
    """Seed-mean e per (train shape, setting) from the arm-S cells."""
    rows = []
    for name in grid.TRAIN:
        d = json.loads((DATA / "cells_S" / f"{name}.json").read_text())
        by = {}
        for r in d["rows"]:
            by.setdefault(skey(r), []).append(r)
        for k, rs in by.items():
            rows.append({"case": name, **{x: rs[0][x] for x in ("nbeta", "q", "c", "gamma", "eps")},
                         "e": float(np.mean([r["e"] for r in rs])), "seeds": len(rs),
                         "ok": all(r["alive"] and r["halves_ok"] for r in rs)})
    return rows


def main():
    tr = train_table()
    trq = [r for r in tr if r["q"] >= 0.1]
    info = {nm: shape_info(nm) for nm in grid.TRAIN}

    # L: a, b by least squares in e (points), TRAIN q >= 0.1
    def resL(p):
        a, b = np.exp(p)
        return [100 * (e_L(info[r["case"]], r, a, b) - r["e"]) for r in trq]
    fL = least_squares(resL, x0=[math.log(1.0), math.log(1.0)])
    aL, bL = (float(v) for v in np.exp(fL.x))

    # L_z: b on TRAIN, all q
    def resZ(p):
        b = math.exp(p[0])
        return [100 * (e_Lz(info[r["case"]], r, b) - r["e"]) for r in tr]
    fZ = least_squares(resZ, x0=[0.0])
    bZ = float(math.exp(fZ.x[0]))

    # nulls
    n1 = float(np.mean([r["e"] for r in trq]))

    def resN2(p):
        a, b = np.exp(p)
        return [100 * (e_N2(r, a, b) - r["e"]) for r in trq]
    fN2 = least_squares(resN2, x0=[0.0, 0.0])
    aN2, bN2 = (float(v) for v in np.exp(fN2.x))
    iso = Iso([r["eps"] * r["nbeta"] / r["gamma"] for r in trq], [r["e"] for r in trq])

    def n4(s, inf):
        eps = cell_eps(inf, s["nbeta"], s["gamma"], s["c"])
        pt = np.log([s["nbeta"], s["gamma"], eps])
        d = np.array([np.linalg.norm(pt - np.log([r["nbeta"], r["gamma"], r["eps"]])) for r in trq])
        m = d.min()
        return float(np.mean([r["e"] for r, dd in zip(trq, d) if dd <= m + 1e-9])), int((d <= m + 1e-9).sum())

    timing = json.loads((DATA / "timing_S.json").read_text())["settings"]
    pred = {}
    for group, names in (("blind", BLIND), ("seen", SEEN), ("control", CONTROL), ("train", grid.TRAIN)):
        for nm in names:
            inf = shape_info(nm)
            for s in settings():
                eps = cell_eps(inf, s["nbeta"], s["gamma"], s["c"])
                X = eps * s["nbeta"] / s["gamma"]
                nn, nties = n4(s, inf)
                pred[f"{nm}|{skey(s)}"] = {
                    "case": nm, "group": group, **s, "eps": eps, "T": timing[skey(s)]["T"],
                    "primary": group == "blind" and s["q"] >= 0.1, "lam": inf["lam"], "lam0": inf["lam0"],
                    "z": None if inf["lam0"] == 0 else s["q"] * inf["MNH0"] / inf["lam0"], "X": X,
                    "reg_term": reg_term(inf, s["nbeta"], s["gamma"], eps),
                    "e_L": e_L(inf, s, aL, bL), "e_Lz": e_Lz(inf, s, bZ), "e_N1": n1, "e_N2": e_N2(s, aN2, bN2),
                    "e_N3": iso(X), "e_N4": nn, "N4_ties": nties}

    # gamma rule: q where L's error crosses -5 points at c = 0.01, per training shape and nbeta
    gamma_rule = []
    for nm in grid.TRAIN:
        inf = info[nm]
        for nb in grid.NBETA:
            f = lambda lq: e_L(inf, {"nbeta": nb, "q": 10 ** lq, "c": 0.01,
                                     "gamma": math.sqrt(nb * grid.K4 / 10 ** lq)}, aL, bL) + 0.05
            lo, hi = -6.0, 8.0
            q5 = 10 ** brentq(f, lo, hi) if f(lo) * f(hi) < 0 else None
            gamma_rule.append({"case": nm, "nbeta": nb, "q_5pct": q5,
                               "gamma_5pct": None if q5 is None else math.sqrt(nb * grid.K4 / q5)})

    # S2 Lau et al. 2308.12108 App. J.1 '1k' DLN recipe mapped to our loss (arm P); S2b parent gate row (train shapes)
    s2 = []
    nb2, g2, eps2 = 8686.0 * 100 / 3, 1.0, 5e-7
    for nm in S2_SHAPES:
        inf = shape_info(nm)
        q2 = nb2 * grid.K4 / g2 ** 2
        s = {"nbeta": nb2, "gamma": g2, "q": q2, "c": eps2 * (nb2 * inf["h_max"] + g2)}
        s2.append({"case": nm, **s, "eps": eps2, "steps": 10000, "kept": "last 10%",
                   "e_L_equilibrium": (reg_term(inf, nb2, g2, eps2) + inf["lam0"] * psi(q2, aL, bL)) / inf["lam"] - 1,
                   "e_Lz_equilibrium": (reg_term(inf, nb2, g2, eps2)
                                        + inf["lam0"] * Phi(q2 * inf["MNH0"] / inf["lam0"], bZ)) / inf["lam"] - 1})
    s2b = []
    nbp, gp = 128 / math.log(128), 10.0
    for nm in grid.TRAIN:
        inf = info[nm]
        qp = nbp * grid.K4 / gp ** 2
        for c in (0.03, 0.1, 0.3):
            eps = cell_eps(inf, nbp, gp, c)
            s2b.append({"case": nm, "nbeta": nbp, "gamma": gp, "q": qp, "c": c, "eps": eps,
                        "e_L_equilibrium": (reg_term(inf, nbp, gp, eps) + inf["lam0"] * psi(qp, aL, bL)) / inf["lam"] - 1,
                        "note": "parent llc-early-alarm-8b gate: n = 128, nbeta = n/ln n, gamma 10; its 100-step "
                                "chains measured -26% to -73% (data/gate/grid.json there)"})

    train_fit = [{**r, "e_L": e_L(info[r["case"]], r, aL, bL), "e_Lz": e_Lz(info[r["case"]], r, bZ)} for r in tr]
    out = {
        "written_before_heldout": True, "prereg_commit": "84d8b1861",
        "fit": {"L": {"a": aL, "b": bL, "train_cells": len(trq), "rms_pts": float(np.sqrt(np.mean(np.square(fL.fun))))},
                "L_z": {"b": bZ, "train_cells": len(tr), "rms_pts": float(np.sqrt(np.mean(np.square(fZ.fun))))},
                "N1": {"e": n1}, "N2": {"a": aN2, "b": bN2}, "N3": {"increasing": bool(iso.increasing),
                                                                    "logX": iso.lx.tolist(), "e": iso.fy.tolist()}},
        "T_by_setting": {k: v["T"] for k, v in timing.items()},
        "gamma_rule": gamma_rule, "S2_lau_recipe": s2, "S2b_parent_gate": s2b,
        "train_fit": train_fit, "cells": pred, "cmd": " ".join(sys.argv)}
    (HERE / "predictions.json").write_text(json.dumps(out, indent=1))
    print(f"L: a={aL:.4g} b={bL:.4g} rms {out['fit']['L']['rms_pts']:.2f} pts on {len(trq)} train cells; "
          f"L_z: b={bZ:.4g} rms {out['fit']['L_z']['rms_pts']:.2f}; N1 {n1:+.3f}; N2 a={aN2:.3g} b={bN2:.3g}; "
          f"N3 increasing={iso.increasing}")
    for g in gamma_rule:
        print("gamma rule", g)
    for s in s2:
        print("S2", s["case"], f"q={s['q']:.3g} c={s['c']:.3f} e_L={s['e_L_equilibrium']:+.3f}")
    print("cells", len(pred), "primary", sum(v["primary"] for v in pred.values()))


if __name__ == "__main__":
    main()
