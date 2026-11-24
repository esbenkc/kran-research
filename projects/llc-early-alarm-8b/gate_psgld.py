"""E4, EXPLORATORY, unscored (room #30013; DEVIATIONS line by Iris before it ran): RMSProp-preconditioned SGLD (Li,
Chen, Carlson, Carin 2016, arXiv 1512.07666; the Timaeus sampling guide's default) on the SAME gate case and budget.
Per step: g = n beta grad L_S(w) + gamma (w - w*); v = alpha v + (1 - alpha) g^2 (alpha 0.99), v_hat = v/(1-alpha^t);
G = 1/(sqrt(v_hat) + a), a = 1e-8; w <- w - (eps/2) G g + sqrt(eps G) N(0, I). LLC = n beta (mean draw loss - L(w*)).
Grid eps {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} x gamma {1, 10}; 3 chain-seed repeats (repeat 0 scored like the gate); the
gate's PASS and selection rules, applied mechanically, reported only as exploratory.

    .venv/bin/python gate_psgld.py      # -> data/gate/e4_psgld.json
"""
import itertools
import json
import math

import numpy as np
import torch

import gate
from sgld import nbeta_of

EPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
GAMMA = (1.0, 10.0)
ALPHA, A_STAB, REPEATS = 0.99, 1e-8, 3


def chain(r, eps, gamma, seed):
    cs = gate.make_case(r)
    X, Y = cs["X"], cs["Y"]
    A = cs["A"].clone().requires_grad_(True)
    B = cs["B"].clone().requires_grad_(True)
    params, w_star = [B, A], [cs["B"], cs["A"]]
    n, nbeta = gate.NROWS, nbeta_of(gate.NROWS)
    g = torch.Generator().manual_seed(seed)
    v = [torch.zeros_like(p) for p in params]
    perm, pos, losses = torch.randperm(n, generator=g), 0, []
    for t in range(1, gate.BURN + gate.DRAWS + 1):
        if pos + gate.BATCH > n:
            perm, pos = torch.randperm(n, generator=g), 0
        rows = perm[pos:pos + gate.BATCH].tolist()
        pos += gate.BATCH
        l = 0.5 * ((Y[rows] - X[rows] @ (B @ A).T) ** 2).sum(1).mean()
        gr = torch.autograd.grad(l, params)
        losses.append(float(l.detach()))
        with torch.no_grad():
            for p, gg, a, vv in zip(params, gr, w_star, v):
                d = nbeta * gg + gamma * (p - a)
                vv.mul_(ALPHA).add_((1 - ALPHA) * d * d)
                G = 1.0 / (torch.sqrt(vv / (1 - ALPHA ** t)) + A_STAB)
                p.add_(-(eps / 2) * G * d + torch.sqrt(eps * G) * torch.randn(p.shape, generator=g, dtype=p.dtype))
        if not math.isfinite(losses[-1]):
            break
    return losses, cs["L_wstar"], nbeta


rows = []
for eps, gamma in itertools.product(EPS, GAMMA):
    for r in gate.RANKS:
        for rep in range(REPEATS):
            ch = [chain(r, eps, gamma, 2 * rep + k) for k in range(gate.CHAINS)]
            draws = [c[0][gate.BURN:] for c in ch]
            L_star, nbeta = ch[0][1], ch[0][2]
            lam = [nbeta * (float(np.mean(d)) - L_star) if len(d) == gate.DRAWS else float("nan") for d in draws]
            rows.append({"r": r, "eps": eps, "gamma": gamma, "rep": rep, "chains": lam,
                         "lambda_hat": float(np.mean(lam)),
                         "bm_se": nbeta * gate.batch_means_se(draws) if all(len(d) == gate.DRAWS for d in draws)
                         else float("nan"), "losses": [c[0] for c in ch]})
table = []
for eps, gamma in itertools.product(EPS, GAMMA):
    cell = {"eps": eps, "gamma": gamma}
    for r in gate.RANKS:
        lam = gate.aoyagi(gate.M, gate.N, gate.H, r)[0]
        rs = sorted([x for x in rows if x["r"] == r and x["eps"] == eps and x["gamma"] == gamma], key=lambda y: y["rep"])
        cell[f"r{r}"] = {"truth": lam, "hat": rs[0]["lambda_hat"], "rel_err": (rs[0]["lambda_hat"] - lam) / lam,
                         "bm_se": rs[0]["bm_se"], "repeat_rel_errs": [(x["lambda_hat"] - lam) / lam for x in rs]}
    cell["sep"] = abs(cell["r8"]["hat"] - cell["r1"]["hat"])
    cell["sep_needed"] = gate.SEP * max(cell["r1"]["bm_se"], cell["r8"]["bm_se"])
    cell["worst_err"] = max(abs(cell["r1"]["rel_err"]), abs(cell["r8"]["rel_err"]))
    cell["pass"] = bool(cell["worst_err"] < gate.TOL and cell["sep"] >= cell["sep_needed"])
    table.append(cell)
    print(f"eps={eps:<7g} gamma={gamma:<5g} r1 {cell['r1']['hat']:8.1f} ({cell['r1']['rel_err']:+.3f}) reps "
          f"{[round(x, 3) for x in cell['r1']['repeat_rel_errs']]} r8 {cell['r8']['hat']:8.1f} "
          f"({cell['r8']['rel_err']:+.3f}) reps {[round(x, 3) for x in cell['r8']['repeat_rel_errs']]} "
          f"sep {cell['sep']:.1f}/{cell['sep_needed']:.1f} {'PASS' if cell['pass'] else ''}", flush=True)
passing = sorted([x for x in table if x["pass"]], key=lambda x: (x["worst_err"], x["eps"]))
json.dump({"note": "EXPLORATORY E4, unscored", "alpha": ALPHA, "a": A_STAB, "rows": rows, "table": table,
           "n_pass": len(passing), "selected": passing[0] if passing else None},
          open(gate.OUT / "e4_psgld.json", "w"))
print("E4 pass", len(passing), "of", len(table), "selected", None if not passing else (passing[0]["eps"], passing[0]["gamma"]))
