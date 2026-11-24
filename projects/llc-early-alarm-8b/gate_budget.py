"""EXPLORATORY, unscored (posted after the gate failed, room #29980): the same RRR case and the same sgld.py, with longer
chains. Question: does the estimator reach Aoyagi's value with enough steps, and how many steps does that take?
Budget = total steps per chain S (burn S/4, draws 3S/4), 2 chains, c in {0.03, 0.1}, gamma in {0.1, 1}; repeats 0-2.

    .venv/bin/python gate_budget.py      # -> data/gate/budget.json
"""
import itertools
import json

import numpy as np

import gate

STEPS = (100, 400, 1600, 6400)
rows = []
for c, gamma, S in itertools.product((0.03, 0.1), (0.1, 1.0), STEPS):
    gate.BURN, gate.DRAWS = S // 4, S - S // 4
    for r in gate.RANKS:
        for rep in range(3):
            x = gate.run_setting(r, c, gamma, rep)
            lam = gate.aoyagi(gate.M, gate.N, gate.H, r)[0]
            rows.append({k: x[k] for k in ("r", "c", "gamma", "rep", "eps", "lambda_hat", "chains", "bm_se",
                                           "mala_accept_mean")} | {"steps": S, "burn": gate.BURN, "draws": gate.DRAWS,
                                                                   "rel_err": (x["lambda_hat"] - lam) / lam})
    cell = [x for x in rows if x["c"] == c and x["gamma"] == gamma and x["steps"] == S]
    print(f"c={c} gamma={gamma} steps={S}: " + "  ".join(
        f"r{r} {np.mean([x['lambda_hat'] for x in cell if x['r'] == r]):7.1f} "
        f"({np.mean([x['rel_err'] for x in cell if x['r'] == r]):+.3f}, acc "
        f"{np.mean([x['mala_accept_mean'] for x in cell if x['r'] == r]):.2f})" for r in gate.RANKS), flush=True)
json.dump({"note": "exploratory, unscored; same case as gate.py", "steps": STEPS, "rows": rows},
          open(gate.OUT / "budget.json", "w"))
