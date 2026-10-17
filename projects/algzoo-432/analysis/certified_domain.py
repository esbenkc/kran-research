"""Iris: where is C1 (2nd argmax with phantom 0) a faithful description of the model?

Domain D_tau = inputs whose top-3 gaps in the phantom-augmented order both exceed tau * rms(x)
(M - S > tau*s and S - T > tau*s). Reports coverage of D_tau and C1's agreement with the model inside and
outside it, plus what C1's misses inside D look like. Dev seed 82, n = 1e6 (never the holdout).
Usage: .venv/bin/python certified_domain.py   (~5 s) -> certified_domain.json
"""
import json
import numpy as np
from residual_common import data, c1, aug_order

X, L, m = data(82, 1_000_000)
o = aug_order(X)
v = np.take_along_axis(np.c_[np.zeros(len(X)), X], o + 1, 1)
s = np.sqrt((X ** 2).mean(1))
g12, g23 = (v[:, 0] - v[:, 1]) / s, (v[:, 1] - v[:, 2]) / s
ok = c1(X) == m
res = {"C1 all": round(float(ok.mean()), 5), "domains": []}
for tau in (0.01, 0.02, 0.05, 0.1, 0.2):
    D = (g12 > tau) & (g23 > tau)
    res["domains"].append(dict(tau=tau, coverage=round(float(D.mean()), 4), agree_in=round(float(ok[D].mean()), 5),
                               agree_out=round(float(ok[~D].mean()), 4)))
D = (g12 > 0.1) & (g23 > 0.1)
e = D & ~ok
npos = (X > 0).sum(1)
res["misses in D_0.1"] = dict(n=int(e.sum()), share_one_positive_input=round(float((npos[e] == 1).mean()), 3),
                             share_S_is_phantom=round(float((o[e, 1] == -1).mean()), 3))
one = npos == 1
res["one positive input"] = dict(share=round(float(one.mean()), 5), C1_agree=round(float(ok[one].mean()), 4))
print(json.dumps(res, indent=1))
json.dump(res, open("certified_domain.json", "w"), indent=1)
