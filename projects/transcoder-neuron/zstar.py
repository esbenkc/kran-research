"""Felix #14440 check: after the NEURON-feature removal push du = -a_f e_f/|e_f|^2, where does the matched neuron's
pre-activation z_{n*} land? And after an ORACLE push it is exactly 0 by construction. If the copy's push leaves
z_{n*} well above GELU's bend (|z| > ~1), the ORACLE vs NEURON gap is GELU's bend, which a ReLU copy never meets.

  env/.venv/bin/python zstar.py -> data/zstar.json
"""
import json

import numpy as np
import pandas as pd
import torch

from common import DEV, GPT2

m = GPT2()
W = torch.tensor(np.load("data/windows.npy"), dtype=torch.long)
out = {}
for L in [8, 9, 10, 11]:
    tc = m.tc(L); b = m.blocks[L]
    U = pd.read_csv(f"data/pushes/L{L}_units.csv")
    P = pd.read_csv(f"data/pushes/L{L}_pushes.csv")
    P = P[(P.group == "NEURON") & (P.pre_ok > 0)].reset_index(drop=True)
    nst = dict(zip(U[U.group == "NEURON"].uid, U[U.group == "NEURON"].same_neuron))
    z0s, z1s = [], []
    with torch.no_grad():
        for s in range(0, len(P), 64):
            d = P.iloc[s:s + 64]
            ids = W[d.window.values].to(DEV)
            _, cap = m.forward(ids, capture={f"xn{L}"}, stop_after=L)
            x = cap[f"xn{L}"][torch.arange(len(d), device=DEV), torch.tensor(d.t.values, device=DEV)]
            f = torch.tensor(d.uid.values, device=DEV)
            e = tc.W_enc[:, f].T
            pre = ((x - tc.b_dec) * e).sum(-1) + tc.b_enc[f]
            du = -pre[:, None] * e / (e * e).sum(-1, keepdim=True)
            n = torch.tensor([int(nst[u]) for u in d.uid.values], device=DEV)
            zf = lambda xx: ((xx * b["ln_2.weight"] + b["ln_2.bias"]) * b["mlp.c_fc.weight"][:, n].T).sum(-1) + b["mlp.c_fc.bias"][n]
            z0s.append(zf(x).cpu()); z1s.append(zf(x + du).cpu())
    z0, z1 = torch.cat(z0s).numpy(), torch.cat(z1s).numpy()
    out[L] = {"n": int(len(z0)), "z_before_median": float(np.median(z0)), "z_after_median": float(np.median(z1)),
              "z_after_p10_p90": [float(np.percentile(z1, 10)), float(np.percentile(z1, 90))],
              "share_after_below_0": float((z1 < 0).mean()), "share_after_in_bend_abs_lt_1": float((np.abs(z1) < 1).mean())}
    print(L, out[L])
json.dump(out, open("data/zstar.json", "w"), indent=1)
