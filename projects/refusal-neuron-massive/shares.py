"""Share of each arm's push energy in the 3 outlier coordinates M (weights only, CPU, no model run).
Run: ../env/.venv/bin/python shares.py -> data/shares.json"""
import json
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).parent
M = [1793, 1999, 1401]
share = np.load(P / "data/phase0/share_L13.npy")  # per L13 neuron, energy share of W_down[:, j] in M
meta = json.loads((P / "data/runs/meta.json").read_text())
J = meta["J"]
g = share[3270]
out = {"A": g, "A_hm": g, "A_zero": g, "A_skip0": g, "A_tok0": g, "B": 0.0, "C": 1.0,
       "N5441pos": share[5441], "N5441neg": share[5441]}
for k, j in enumerate(J):
    out[f"Dp_{k}"] = out[f"Ds_{k}"] = 1.0
    out[f"Ep_{k}"] = 0.0
    out[f"Fpos_{k}"] = out[f"Fneg_{k}"] = share[j]
gen = torch.Generator().manual_seed(0)  # same draws as arms.py (randn(2048) x 20, after nothing else)
for k in range(20):
    u = torch.randn(2048, generator=gen)
    u = u / u.norm()
    out[f"Fp_{k}"] = float((u[M] ** 2).sum())
(P / "data/shares.json").write_text(json.dumps({k: float(v) for k, v in out.items()}, indent=1))
print("gate", g, "N5441", share[5441], "F median", np.median([share[j] for j in J]),
      "Fp median", np.median([out[f'Fp_{k}'] for k in range(20)]))
