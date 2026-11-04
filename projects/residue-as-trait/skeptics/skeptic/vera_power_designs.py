"""D_add power (frozen s3 rule) for the cost-cut design options after smoke (#14170)."""
import numpy as np, json
from vera_power_v21 import world
src = open("vera_power_add.py").read().split("rng = np.random.default_rng(5)")[0]
ns = {}; exec(src, ns); dadd, null = ns["dadd"], ns["null"]
rng = np.random.default_rng(1170); out = {}
for T, S in ((8, 5), (7, 5), (8, 3), (6, 4), (4, 5)):
    for name, kw in (("sd1.14", dict(s_int=1.14)), ("sd1.5", dict(s_int=1.5)), ("sd2.5", dict(s_int=2.5)),
                     ("null sd0", dict(s_int=0, trait=False)), ("null sd1.5", dict(s_int=1.5, trait=False))):
        lab = []
        for _ in range(300):
            y = world(rng, T, S, tau=1.0, **kw); d = dadd(y)
            fl = y.sum() < 0.08 * y.size or (y.sum(axis=(0, 2, 3)) > 0).sum() < 3
            q = np.quantile(null(y, rng, 300), .95)
            lab.append("U" if fl else "Y" if (d >= .02 and d > q) else "N" if d <= .005 else "U")
        lab = np.array(lab); k = f"{T}x{S}={T*S*10} {name}"
        out[k] = {c: float((lab == c).mean()) for c in "YNU"}
        print(f"{k:24s} YES {out[k]['Y']:.2f} NO {out[k]['N']:.2f} UNC {out[k]['U']:.2f}", flush=True)
json.dump(out, open("vera_power_designs.json", "w"), indent=1)
