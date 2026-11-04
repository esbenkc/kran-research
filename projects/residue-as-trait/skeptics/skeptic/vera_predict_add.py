"""Predicted D_add (frozen formula, vera_power_add.dadd) for v1.1.
Anchored: agent x scenario sd 1.14 (parent agent x arm), scenario sd 1.0. Prior: sd ~ U(0.5, 3.0), tau ~ U(0.5, 1.5)."""
import numpy as np, json, importlib.util, sys
sys.argv = ["x"]
from vera_power_v21 import world
src = open("vera_power_add.py").read().split("rng = np.random.default_rng(5)")[0]
ns = {}; exec(src, ns); dadd, null = ns["dadd"], ns["null"]
rng = np.random.default_rng(41104); out = {}
def go(label, draw, n=600):
    ds, lab = [], []
    for _ in range(n):
        sd, tau = draw()
        y = world(rng, 8, 5, s_int=sd, tau=tau)
        d = dadd(y); q = np.quantile(null(y, rng, 200), .95)
        ds.append(d); lab.append("Y" if (d >= .02 and d > q) else "N" if d <= .005 else "U")
    ds = np.array(ds); lab = np.array(lab)
    out[label] = dict(D_med=float(np.median(ds)), q10=float(np.quantile(ds, .1)), q90=float(np.quantile(ds, .9)),
                      P_yes=float((lab == "Y").mean()), P_no=float((lab == "N").mean()))
    print(label, out[label], flush=True)
go("anchored sd1.14 tau1.0", lambda: (1.14, 1.0))
go("prior sd U(0.5,3) tau U(0.5,1.5)", lambda: (rng.uniform(0.5, 3.0), rng.uniform(0.5, 1.5)))
json.dump(out, open("vera_predict_add.json", "w"), indent=1)
