"""llc-bias-law runs (Bruno). Every number in the post comes from the JSON files this script writes under data/.

    .venv/bin/python run.py timing --arm S            # frozen T rule on the TRAIN cases, seed 0 -> data/timing_S.json
    .venv/bin/python run.py cells  --arm S --cases 32x32x16r1 32x32x16r8 --seeds 0 1 2   -> data/cells_S/<case>.json
    .venv/bin/python run.py gibbs  --arm S --cases ... --seeds 0 1 2                     -> data/gibbs_S/<case>.json

arm S = minibatch SGLD on n = 1024 noise-free rows, batch 64 (y = W0 x exactly, so w* is an exact minimum);
arm P = full-gradient Langevin on the population loss K(w) = 1/2 ||BA - W0||_F^2 (x ~ N(0, I) integrated out).
Case name 'MxNxHrR'. Seed s = teacher draw s (+ data draw s in arm S) + chain noise stream s.
Grid constants live in grid.py (copied from PREREG.md's FROZEN RULE).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import grid
from core import SGLD, data, gibbs, hessian_eigs, teacher

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "research"))
from rrr_lambda import rrr_lambda  # noqa: E402

DATA = HERE / "data"


def parse_case(name):
    shape, r = name.split("r")
    M, N, H = (int(v) for v in shape.split("x"))
    return M, N, H, int(r)


def case_setup(name, seed, arm):
    """Teacher, data, curvature for one (case, seed). Returns dict of tensors + h (positive Hessian eigenvalues)."""
    M, N, H, r = parse_case(name)
    A, B = teacher(M, N, H, r, seed)
    if arm == "S":
        X = data(M, grid.N_ROWS, seed)
        S = X.T @ X / grid.N_ROWS
    else:
        X, S = None, torch.eye(M, dtype=torch.float64)
    # PREREG 2(1): q, z and h_i (and h_max in eps) use the POPULATION Hessian of K at w* (S = I), in both arms
    ev = hessian_eigs(A, B, torch.eye(M, dtype=torch.float64)).numpy()
    h = ev[ev > 1e-8 * ev.max()]
    ev_n = hessian_eigs(A, B, S).numpy()
    lam = float(rrr_lambda(M, N, H, r))
    lam0 = float(rrr_lambda(M - r, N - r, H - r, 0))
    return {"name": name, "M": M, "N": N, "H": H, "r": r, "A": A, "B": B, "X": X, "S": S, "h": h,
            "h_max": float(ev.max()), "h_max_emp": float(ev_n.max()), "lam": lam, "lam0": lam0, "d_reg": r * (M + N - r)}


def settings():
    """Every (nbeta, q, c) of the frozen grid with gamma = sqrt(nbeta*k4/q)."""
    out = []
    for nb in grid.NBETA:
        for q in grid.Q:
            for c in grid.C:
                out.append({"nbeta": nb, "q": q, "c": c, "gamma": math.sqrt(nb * grid.K4 / q)})
    return out


def skey(s):
    return f"nb{s['nbeta']:g}_q{s['q']:g}_c{s['c']:g}"


def make_chains(cases, sets, arm, seed, block, chains=1):
    """Stack `chains` chains per (case setup, setting) into one SGLD batch, order (case, setting, chain)."""
    A0, B0, X, S, eps, gam, nb = [], [], [], [], [], [], []
    for cs in cases:
        for s in sets:
            for _ in range(chains):
                A0.append(cs["A"]); B0.append(cs["B"]); X.append(cs["X"]); S.append(cs["S"])
                eps.append(s["c"] / (s["nbeta"] * cs["h_max"] + s["gamma"]))
                gam.append(s["gamma"]); nb.append(s["nbeta"])
    Xs = None if arm == "P" else torch.stack(X)
    ch = SGLD(torch.stack(A0), torch.stack(B0), Xs, torch.stack(S), eps, gam, nb, grid.BATCH, seed, block)
    return ch, np.array(eps)


def cmd_timing(a):
    """Frozen T rule (PREREG 2(5)): per setting, the smallest T in grid.T_GRID with |e(T) - e(3T)| < 1 point on BOTH
    training cases at seed 0 (burn-in = half the chain); else grid.T_FALLBACK. One chain of length 3*T_max per
    (case, setting) gives every e(T) and e(3T) (prefix windows of the same chain)."""
    t0 = time.time()
    sets = settings()
    cases = [case_setup(nm, 0, a.arm) for nm in grid.TRAIN]
    block = grid.BLOCK
    res = {s_k: {"setting": s, "e": {}, "T": None} for s_k, s in ((skey(s), s) for s in sets)}
    active = list(range(len(sets)))
    for Tg in grid.T_GRID:
        L = 3 * Tg
        if not active:
            break
        # one seed-0 chain of length 3*Tg per (training shape, still-active setting), started at w*
        act_sets = [sets[i] for i in active]
        ch, _ = make_chains(cases, act_sets, a.arm, seed=0, block=block)
        bl = ch.run(L)
        nA = len(act_sets)
        still = []
        for j, i in enumerate(active):
            s = sets[i]
            k = skey(s)
            ok_all = True
            for ci, cs in enumerate(cases):
                row = bl[ci * nA + j]
                eT = float(s["nbeta"] * row[(Tg // 2) // block:Tg // block].mean() / cs["lam"] - 1)
                e3T = float(s["nbeta"] * row[(3 * Tg // 2) // block:L // block].mean() / cs["lam"] - 1)
                res[k]["e"].setdefault(cs["name"], []).append({"T": Tg, "e_T": eT, "e_3T": e3T,
                                                               "diff_pts": 100 * abs(eT - e3T)})
                ok_all &= bool(np.isfinite(eT) and np.isfinite(e3T) and 100 * abs(eT - e3T) < grid.T_TOL_PTS)
            if ok_all:
                res[k]["T"] = Tg
            else:
                still.append(i)
        active = still
        print(f"T={Tg}: {len(sets) - len(active)}/{len(sets)} settings fixed, {time.time() - t0:.0f}s", flush=True)
    for i in active:
        res[skey(sets[i])]["T"] = grid.T_FALLBACK
        res[skey(sets[i])]["fallback"] = True
    out = {"arm": a.arm, "rule": "smallest T in T_GRID with |e(T)-e(3T)| < %g pts on both TRAIN cases, seed 0, "
           "burn-in T/2; else T_FALLBACK" % grid.T_TOL_PTS, "T_GRID": grid.T_GRID, "T_FALLBACK": grid.T_FALLBACK,
           "train": grid.TRAIN, "secs": time.time() - t0, "settings": res,
           "cmd": " ".join(sys.argv)}
    DATA.mkdir(exist_ok=True)
    (DATA / f"timing_{a.arm}.json").write_text(json.dumps(out, indent=1))
    print("wrote", DATA / f"timing_{a.arm}.json")


def cmd_cells(a):
    timing = json.loads((DATA / f"timing_{a.arm}.json").read_text())["settings"]
    sets = settings() if not a.only else [s for s in settings() if skey(s) in a.only]
    block, nch = grid.BLOCK, grid.CHAINS
    outdir = DATA / (a.out or f"cells_{a.arm}")
    outdir.mkdir(parents=True, exist_ok=True)
    for name in a.cases:
        t0 = time.time()
        cases = [case_setup(name, sd, a.arm) for sd in a.seeds]
        rows, blocks_store = [], {}
        Ts = sorted({timing[skey(s)]["T"] for s in sets})
        for T in Ts:
            grp = [s for s in sets if timing[skey(s)]["T"] == T]
            for cs, sd in zip(cases, a.seeds):          # one batch per seed: chain-noise seed = seed
                ch, eps = make_chains([cs], grp, a.arm, seed=sd, block=block, chains=nch)
                bl = ch.run(T)
                d2 = ch.dist2()
                burn = T // 2
                mid = burn + (T - burn) // 2
                for j, s in enumerate(grp):
                    b = bl[j * nch:(j + 1) * nch]                       # (chains, blocks)
                    f = s["nbeta"] / cs["lam"]
                    e_ch = f * b[:, burn // block:].mean(1) - 1
                    e = float(f * b[:, burn // block:].mean() - 1)
                    e1 = float(f * b[:, burn // block:mid // block].mean() - 1)
                    e2 = float(f * b[:, mid // block:].mean() - 1)
                    alive = bool(np.isfinite(b).all() and np.all(np.abs(e_ch) <= grid.DIVERGE_E))
                    rows.append({"case": name, "M": cs["M"], "N": cs["N"], "H": cs["H"], "r": cs["r"],
                                 "lam": cs["lam"], "lam0": cs["lam0"], "d_reg": cs["d_reg"], "seed": sd,
                                 **s, "eps": float(eps[j * nch]), "h_max": cs["h_max"],
                                 "h_max_emp": cs["h_max_emp"], "T": T, "burn": burn, "chains": nch,
                                 "e": e, "e_chains": [float(x) for x in e_ch], "e_half1": e1, "e_half2": e2,
                                 "halves_ok": bool(abs(e1 - e2) * 100 <= grid.HALVES_PTS), "alive": alive,
                                 "dist2_end": [float(x) for x in d2[j * nch:(j + 1) * nch]]})
                    blocks_store[f"{skey(s)}_s{sd}"] = b.astype(np.float32)
            print(f"{name} T={T}: {len(grp)} settings x {len(a.seeds)} seeds x {nch} chains, {time.time() - t0:.0f}s",
                  flush=True)
        hs = {str(sd): cs["h"].tolist() for cs, sd in zip(cases, a.seeds)}
        out = {"case": name, "arm": a.arm, "block": block, "n_rows": grid.N_ROWS if a.arm == "S" else None,
               "batch": grid.BATCH if a.arm == "S" else None, "seeds": a.seeds, "hessian_pos_eigs_pop": hs,
               "rows": rows, "secs": time.time() - t0, "cmd": " ".join(sys.argv)}
        (outdir / f"{name}.json").write_text(json.dumps(out))
        np.savez_compressed(outdir / f"{name}_blocks.npz", **blocks_store)
        print("wrote", outdir / f"{name}.json", f"{time.time() - t0:.0f}s", flush=True)


def cmd_gibbs(a):
    outdir = DATA / f"gibbs_{a.arm}"
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = sorted({(s["nbeta"], s["q"], s["gamma"]) for s in settings()})
    for name in a.cases:
        t0 = time.time()
        rows = []
        for sd in a.seeds:
            cs = case_setup(name, sd, a.arm)
            K = len(pairs)
            A0 = cs["A"].expand(K, *cs["A"].shape).clone()
            B0 = cs["B"].expand(K, *cs["B"].shape).clone()
            S = cs["S"].expand(K, *cs["S"].shape).clone()
            bl = gibbs(A0, B0, S, [p[2] for p in pairs], [p[0] for p in pairs], grid.GIBBS_SWEEPS, seed=sd,
                       block=grid.GIBBS_BLOCK)
            nb_ = grid.GIBBS_BURN // grid.GIBBS_BLOCK
            half = nb_ + (bl.shape[1] - nb_) // 2
            for j, (nb, q, gm) in enumerate(pairs):
                rows.append({"case": name, "seed": sd, "nbeta": nb, "q": q, "gamma": gm, "lam": cs["lam"],
                             "e_gibbs": float(nb * bl[j, nb_:].mean() / cs["lam"] - 1),
                             "e_half1": float(nb * bl[j, nb_:half].mean() / cs["lam"] - 1),
                             "e_half2": float(nb * bl[j, half:].mean() / cs["lam"] - 1)})
        out = {"case": name, "arm": a.arm, "sweeps": grid.GIBBS_SWEEPS, "burn": grid.GIBBS_BURN, "rows": rows,
               "secs": time.time() - t0, "cmd": " ".join(sys.argv)}
        (outdir / f"{name}.json").write_text(json.dumps(out))
        print("wrote", outdir / f"{name}.json", f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timing", "cells", "gibbs"])
    ap.add_argument("--arm", choices=["S", "P"], default="S")
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=[], help="restrict cells to these setting keys")
    ap.add_argument("--out", default=None, help="output subdir of data/ (default cells_<arm>)")
    a = ap.parse_args()
    torch.set_num_threads(grid.THREADS)
    {"timing": cmd_timing, "cells": cmd_cells, "gibbs": cmd_gibbs}[a.cmd](a)


if __name__ == "__main__":
    main()
