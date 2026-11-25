"""Iris: exact block Gibbs (no step size, no chain length) at the parent gate's spring, gamma 10, nbeta 128/ln 128,
on the TRAIN shapes only, noise-free arm-S data (n = 1024) and population loss, 2 seeds each.
Output: data/s2b/gibbs_parent_row.json. Compared with predictions.json S2b_parent_gate (law L equilibrium)."""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)
from run import case_setup
from core import gibbs
import grid

NB, GAMMA = 128 / math.log(128), 10.0
rows = []
for name in grid.TRAIN:
    for arm in ("S", "P"):
        for sd in (0, 1):
            cs = case_setup(name, sd, arm)
            A0, B0, S = cs["A"].unsqueeze(0), cs["B"].unsqueeze(0), cs["S"].unsqueeze(0)
            bl = gibbs(A0.repeat(4, 1, 1), B0.repeat(4, 1, 1), S.repeat(4, 1, 1), [GAMMA] * 4, [NB] * 4,
                       grid.GIBBS_SWEEPS, seed=1000 + sd, block=grid.GIBBS_BLOCK)
            nb_ = grid.GIBBS_BURN // grid.GIBBS_BLOCK
            e_ch = NB * bl[:, nb_:].mean(1) / cs["lam"] - 1
            rows.append({"case": name, "arm": arm, "seed": sd, "nbeta": NB, "gamma": GAMMA,
                         "q": NB * grid.K4 / GAMMA ** 2, "e_chains": e_ch.tolist(), "e": float(e_ch.mean())})
            print(name, arm, sd, "e = %.2f%%" % (100 * e_ch.mean()), "chains", np.round(100 * e_ch, 2))
pred = {(p["case"], p["c"]): p["e_L_equilibrium"] for p in json.load(open("predictions.json"))["S2b_parent_gate"]}
os.makedirs("data/s2b", exist_ok=True)
json.dump({"rows": rows, "law_L_equilibrium": {f"{k[0]} c{k[1]}": v for k, v in pred.items()},
           "sweeps": grid.GIBBS_SWEEPS, "burn": grid.GIBBS_BURN}, open("data/s2b/gibbs_parent_row.json", "w"), indent=1)
