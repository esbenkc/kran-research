"""Tove #14439 (3): why do pushes along a REAL neuron's input direction (ORACLE) score worse than its copies?
Hypothesis: most GPT-2 neurons have no NEURON-class copy in the transcoder, so the dictionary cannot answer a push
that turns such a neuron off. Split ORACLE R^2 by whether the pushed neuron has a NEURON-class copy (enc and dec
|cos| >= 0.9, the PREREG rule) in the full 24,576-feature dictionary.

  env/.venv/bin/python oracle_copy.py -> data/oracle_copy.json
"""
import json

import numpy as np
import pandas as pd
import torch

from common import GPT2

m = GPT2()
out = {}
r2 = lambda p, y: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
for L in [8, 9, 10, 11]:
    tc = m.tc(L); b = m.blocks[L]
    with torch.no_grad():
        Win = b["ln_2.weight"][:, None] * b["mlp.c_fc.weight"]; Win = Win - Win.mean(0, keepdim=True)
        Win = Win / Win.norm(dim=0, keepdim=True)
        Wo = b["mlp.c_proj.weight"] - b["mlp.c_proj.weight"].mean(1, keepdim=True); Wo = Wo / Wo.norm(dim=1, keepdim=True)
        copies = torch.zeros(3072, dtype=torch.long)
        for s in range(0, 24576, 2048):
            E = tc.W_enc[:, s:s + 2048]; E = E - E.mean(0, keepdim=True); E = E / E.norm(dim=0, keepdim=True)
            D = tc.W_dec[s:s + 2048]; D = D - D.mean(1, keepdim=True); D = D / D.norm(dim=1, keepdim=True)
            ce = (E.T @ Win).abs(); v, i = ce.max(1)
            dc = (D @ Wo.T).abs().gather(1, i[:, None]).squeeze(1)
            neu = (v >= 0.9) & (dc >= 0.9)
            copies += torch.bincount(i[neu].cpu(), minlength=3072)
    P = pd.read_csv(f"data/pushes/L{L}_pushes.csv")
    P = P[(P.group == "ORACLE") & (P.pre_ok > 0)]
    has = P.uid.map(lambda n: int(copies[n]) > 0)
    out[L] = {"neurons_with_copy": int((copies > 0).sum()), "neurons": 3072,
              "oracle_pushes_with_copy": int(has.sum()), "oracle_pushes_without": int((~has).sum()),
              "R2_with_copy": r2(P.total_r[has].values, P.meas_local_r[has].values) if has.sum() > 10 else None,
              "R2_without_copy": r2(P.total_r[~has].values, P.meas_local_r[~has].values) if (~has).sum() > 10 else None,
              "cos_vec_with": float(P.cos_vec[has].median()) if has.sum() else None,
              "cos_vec_without": float(P.cos_vec[~has].median()) if (~has).sum() else None,
              "curv_with": float(P.curv[has].median()) if has.sum() else None,
              "curv_without": float(P.curv[~has].median()) if (~has).sum() else None}
    print(L, out[L])
json.dump(out, open("data/oracle_copy.json", "w"), indent=1)
