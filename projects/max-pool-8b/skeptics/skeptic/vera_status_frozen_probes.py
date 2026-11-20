"""Vera: the frozen status (PREREG s6) with the BUILD's frozen monitors (scores.npz = rows . probes.npz w + b, checked
to <= 5e-7 in vera_rebuild.py), scored states from my frozen-rule picks; my own city bootstrap and t-interval."""
import sys, json
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vera_core as vc
D = Path(sys.argv[1]); M = vc.meta(D / "ref/split.json")
R = json.load(open("vera_rebuild.json")); S = sorted(v for k, v in R["scored"].items() if k.startswith("main__"))
out = {"scored": S}
for k in ("max", "mean", "last", "cty"):
    L = [np.load(D / "states" / s / "scores.npz")[k].astype(np.float64) for s in S]
    pts = [vc.catch(x, M["y"], M["split"]) for x in L]
    b = vc.city_boot(L, M["y"], M["split"], M["city"], n=2000, seed=0)
    out[k] = dict(per_seed=[p["catch"] for p in pts], fa=[p["fa"] for p in pts], pooled=float(np.mean([p["catch"] for p in pts])),
                  ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], seed_t=vc.t_interval([p["catch"] for p in pts]), _b=b)
for o in ("mean", "last", "cty"):
    d = out["max"]["_b"] - out[o]["_b"]
    out[f"max-{o}"] = dict(diff=out["max"]["pooled"] - out[o]["pooled"], ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
for k in ("max", "mean", "last", "cty"):
    out[k].pop("_b")
pm = out["max"]; lo, hi = pm["ci"]
out["status"] = "CONFIRMED" if (pm["pooled"] >= .5 and lo >= .4) else ("REJECTED" if hi < .5 else "UNCONFIRMED")
k = sum(c < .2 for c in pm["per_seed"]); ci = stats.binomtest(k, 3).proportion_ci(method="exact")
out["prediction"] = dict(k_below_20=k, n=3, cp95=[ci.low, ci.high], outcome="held" if k == 3 else "failed")
print(json.dumps(out, indent=1, default=float))
Path("vera_status_frozen_probes.json").write_text(json.dumps(out, indent=1, default=float))
