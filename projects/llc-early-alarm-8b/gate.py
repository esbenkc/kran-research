"""Phase-0 gate (PREREG FROZEN RULE (9)): the frozen SGLD estimator (sgld.py) must recover Aoyagi's closed-form
learning coefficient of a LoRA-shaped reduced-rank regression within 10% at true ranks 1 and 8, AND separate the two
ranks: |lambda_8 - lambda_1| >= 3 x max(batch-means SE at rank 1, at rank 8), at the 8B budget (2 chains x (25 burn
+ 75 draws), minibatch 8 of n = 128 rows, n*beta = n / ln n = 26.38) and the 8B step rule eps = c/(n beta h_max + gamma).

Model y = B A x + e: A in R^{H x M}, B in R^{N x H}, M = N = 32, H = 16. Data (fixed numpy seeds): x ~ N(0, I_32)
(seed 1), e ~ N(0, I_32) (seed 2); the true B A = U V^T has rank r with all nonzero singular values 1, U, V random
orthonormal (seed 0, QR of a Gaussian). True parameter: B[:, :r] = U, A[:r] = V^T, all other entries 0.
Loss = Gaussian NLL with the true variance, constants dropped: L(w) = (1/2) mean_i ||y_i - B A x_i||^2.
w* = the empirical minimum: full-batch Adam (lr 1e-3) started at the true parameter until ||grad L|| < 1e-6.
h_max = the largest eigenvalue of the exact Hessian of L at w* (eigh). Acceptance diagnostic (Lau et al. 2308.12108,
App. H.1): the MALA acceptance probability of every SGLD step on the full-batch loss is logged (nothing is rejected).
The scored estimate per (setting, rank) is repeat 0 (chain seeds 0, 1); repeats 1-4 (other chain seeds) are reported
as a spread diagnostic only.

    .venv/bin/python gate.py --grid       # every setting of the posted grid -> data/gate/grid.json
    .venv/bin/python gate.py --select     # the posted selection rule -> data/gate/frozen.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from sgld import nbeta_of, sgld_chain, step_size

HERE = Path(__file__).parent
OUT = HERE / "data" / "gate"
M = N = 32
H = 16
RANKS = (1, 8)
NROWS = 128
BATCH = 8
CHAINS = 2
BURN, DRAWS = 25, 75
REPEATS = 5
C_GRID = (0.03, 0.1, 0.3, 1.0)
GAMMA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
TOL, SEP = 0.10, 3.0


def aoyagi(M: int, N: int, H: int, r: int):
    """Aoyagi & Watanabe (2005) RLCT lambda and multiplicity m of reduced-rank regression (input M, output N,
    model rank H, true rank r)."""
    if M + N < H + r:
        return M * N / 2, 1
    if M + H < N + r:
        return (H * M - H * r + N * r) / 2, 1
    if N + H < M + r:
        return (H * N - H * r + M * r) / 2, 1
    q = 2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2
    if (M + H + N + r) % 2 == 0:
        return q / 8, 1
    return (q + 1) / 8, 2


def batch_means_se(chain_draws, n_batches=5):
    """PREREG (3): batch-means SE of the mean over the pooled draws (chain 0 then chain 1), 5 batches."""
    x = np.concatenate([np.asarray(c, float) for c in chain_draws])
    b = x[: len(x) // n_batches * n_batches].reshape(n_batches, -1).mean(1)
    return float(b.std(ddof=1) / np.sqrt(n_batches))


_CASES = {}


def make_case(r: int):
    if r in _CASES:
        return _CASES[r]
    g0, g1, g2 = (np.random.default_rng(s) for s in (0, 1, 2))
    U, _ = np.linalg.qr(g0.standard_normal((N, r)))
    V, _ = np.linalg.qr(g0.standard_normal((M, r)))
    X = g1.standard_normal((NROWS, M))
    E = g2.standard_normal((NROWS, N))
    Y = X @ (U @ V.T).T + E
    A = np.zeros((H, M))
    B = np.zeros((N, H))
    A[:r], B[:, :r] = V.T, U
    X, Y, A, B = (torch.tensor(v, dtype=torch.float64) for v in (X, Y, A, B))
    # w* = the empirical minimum: full-batch Adam from the true parameter until ||grad L|| < 1e-6
    A.requires_grad_(True)
    B.requires_grad_(True)
    opt = torch.optim.Adam([A, B], lr=1e-3)
    it, gn = 0, float("inf")
    while gn >= 1e-6 and it < 200_000:
        opt.zero_grad()
        l = 0.5 * ((Y - X @ (B @ A).T) ** 2).sum(1).mean()
        l.backward()
        gn = float(torch.sqrt((A.grad ** 2).sum() + (B.grad ** 2).sum()))
        opt.step()
        it += 1
    A, B = A.detach(), B.detach()
    w = torch.cat([B.reshape(-1), A.reshape(-1)])

    def f(wv):
        Bv = wv[:N * H].reshape(N, H)
        Av = wv[N * H:].reshape(H, M)
        return 0.5 * ((Y - X @ (Bv @ Av).T) ** 2).sum(1).mean()

    h_max = float(torch.linalg.eigvalsh(torch.autograd.functional.hessian(f, w))[-1])
    case = {"X": X, "Y": Y, "A": A, "B": B, "h_max": h_max, "adam_iters": it, "grad_norm_wstar": gn,
            "L_wstar": float(f(w)), "rank_wstar": int(torch.linalg.matrix_rank(B @ A, tol=1e-6))}
    _CASES[r] = case
    return case


def run_setting(r: int, c: float, gamma: float, rep: int):
    cs = make_case(r)
    X, Y = cs["X"], cs["Y"]
    A = cs["A"].clone().requires_grad_(True)
    B = cs["B"].clone().requires_grad_(True)
    params, w_star = [B, A], [cs["B"], cs["A"]]
    L_star = cs["L_wstar"]

    def loss_rows(rows):
        return 0.5 * ((Y[rows] - X[rows] @ (B @ A).T) ** 2).sum(1).mean()

    full = lambda: loss_rows(list(range(NROWS)))
    nbeta = nbeta_of(NROWS)
    eps = step_size(c, nbeta, cs["h_max"], gamma)
    chains = [sgld_chain(loss_rows, params, w_star, NROWS, BATCH, nbeta, eps, gamma, BURN, DRAWS,
                         seed=2 * rep + k, full_loss_fn=full, mala=True) for k in range(CHAINS)]
    draws = [ch["losses"][BURN:] for ch in chains]
    lam_chain = [nbeta * (float(np.mean(d)) - L_star) for d in draws]
    return {"r": r, "c": c, "gamma": gamma, "rep": rep, "eps": eps, "nbeta": nbeta, "h_max": cs["h_max"],
            "L_wstar": L_star, "chains": lam_chain, "lambda_hat": float(np.mean(lam_chain)),
            "bm_se": nbeta * batch_means_se(draws),
            "mala_accept_mean": float(np.mean([np.mean(ch["mala_accept"]) for ch in chains])),
            "dist_from_wstar": [ch["dist_from_wstar"] for ch in chains], "losses": [ch["losses"] for ch in chains]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if a.grid:
        t0 = time.time()
        rows = []
        for c, gamma in itertools.product(C_GRID, GAMMA_GRID):
            for r in RANKS:
                for rep in range(REPEATS):
                    rows.append(run_setting(r, c, gamma, rep))
            print(c, gamma, {r: round(next(x["lambda_hat"] for x in rows if x["r"] == r and x["c"] == c and
                                           x["gamma"] == gamma and x["rep"] == 0), 1) for r in RANKS},
                  round(time.time() - t0, 1), "s", flush=True)
        cases = {str(r): {k: v for k, v in make_case(r).items() if k not in ("X", "Y", "A", "B")} for r in RANKS}
        res = {"M": M, "N": N, "H": H, "ranks": RANKS, "n": NROWS, "batch": BATCH, "chains": CHAINS, "burn": BURN,
               "draws": DRAWS, "repeats": REPEATS, "c_grid": C_GRID, "gamma_grid": GAMMA_GRID,
               "truth": {str(r): dict(zip(("lambda", "m"), aoyagi(M, N, H, r))) for r in RANKS}, "cases": cases,
               "rows": rows, "secs": time.time() - t0}
        (OUT / "grid.json").write_text(json.dumps(res))
    if a.select:
        g = json.load(open(OUT / "grid.json"))
        table = []
        for c, gamma in itertools.product(g["c_grid"], g["gamma_grid"]):
            cell = {"c": c, "gamma": gamma}
            for r in g["ranks"]:
                rs = [x for x in g["rows"] if x["r"] == r and x["c"] == c and x["gamma"] == gamma]
                x0 = next(x for x in rs if x["rep"] == 0)
                lam = g["truth"][str(r)]["lambda"]
                cell[f"r{r}"] = {"truth": lam, "hat": x0["lambda_hat"], "rel_err": (x0["lambda_hat"] - lam) / lam,
                                 "bm_se": x0["bm_se"], "chains": x0["chains"], "mala_accept": x0["mala_accept_mean"],
                                 "eps": x0["eps"], "h_max": x0["h_max"],
                                 "repeat_hats": [x["lambda_hat"] for x in sorted(rs, key=lambda y: y["rep"])],
                                 "repeat_rel_errs": [(x["lambda_hat"] - lam) / lam for x in sorted(rs, key=lambda y: y["rep"])]}
            r1, r2 = (f"r{r}" for r in g["ranks"])
            cell["sep"] = abs(cell[r2]["hat"] - cell[r1]["hat"])
            cell["sep_needed"] = SEP * max(cell[r1]["bm_se"], cell[r2]["bm_se"])
            cell["worst_err"] = max(abs(cell[r1]["rel_err"]), abs(cell[r2]["rel_err"]))
            cell["pass"] = bool(cell["worst_err"] < TOL and cell["sep"] >= cell["sep_needed"])
            table.append(cell)
        passing = sorted([x for x in table if x["pass"]], key=lambda x: (x["worst_err"], x["c"]))
        out = {"rule": "PASS = |rel err| < 10% at ranks 1 and 8 (repeat 0) and |lambda_8 - lambda_1| >= 3 x max(batch-"
                       "means SE); among passing settings: the smallest worst-case error over the 2 ranks; ties: the "
                       "smaller c", "table": table, "n_pass": len(passing),
               "selected": passing[0] if passing else None, "gate": "PASS" if passing else "FAIL"}
        (OUT / "frozen.json").write_text(json.dumps(out, indent=1))
        for x in table:
            print(f"c={x['c']:<5} gamma={x['gamma']:<7} r1 {x['r1']['hat']:8.1f} ({x['r1']['rel_err']:+.3f}) "
                  f"r8 {x['r8']['hat']:8.1f} ({x['r8']['rel_err']:+.3f}) sep {x['sep']:6.1f}/{x['sep_needed']:5.1f} "
                  f"acc {x['r1']['mala_accept']:.2f}/{x['r8']['mala_accept']:.2f} reps r1 "
                  f"[{min(x['r1']['repeat_rel_errs']):+.2f},{max(x['r1']['repeat_rel_errs']):+.2f}] r8 "
                  f"[{min(x['r8']['repeat_rel_errs']):+.2f},{max(x['r8']['repeat_rel_errs']):+.2f}] "
                  f"{'PASS' if x['pass'] else ''}")
        print("GATE", out["gate"], "selected", None if not passing else (passing[0]["c"], passing[0]["gamma"]))


if __name__ == "__main__":
    main()
