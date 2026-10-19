"""City-cluster paired bootstrap (PREREG 'Uncertainty'): 2000 draws, rng 0; resample the 100 refit cities and the
298 eval cities with replacement; re-pick the threshold in every draw; same draw for every seed and monitor.
Input: vera_scores.npz from vera_rebuild.py. Own code."""
import json
from pathlib import Path

import numpy as np

from vera_core import meta, thr

HERE = Path(__file__).parent
M = meta()
y, split, city = M["y"], M["split"], M["city"]
Z = np.load(HERE / "vera_scores.npz")
seeds = sorted({"__".join(k.split("__")[:3]) for k in Z.files if k.startswith("lam0")})
mons = ["frozen", "max14", "mean8", "fresh", "last14"]
rc = np.unique(city[split == "refit"])
ec = np.unique(city[split == "eval"])
# per city: indices of its refit TRUE statements / eval FALSE statements
rneg = {c: np.where((city == c) & (split == "refit") & (y == 0))[0] for c in rc}
epos = {c: np.where((city == c) & (split == "eval") & (y == 1))[0] for c in ec}
assert sum(len(v) for v in rneg.values()) == 200 and sum(len(v) for v in epos.values()) == 596


def pooled(ir, ip):
    out = {}
    for m in mons:
        out[m] = np.mean([(Z[f"{s}__{m}"][ip] > thr(Z[f"{s}__{m}"][ir])).mean() for s in seeds])
    return out


point = pooled(np.concatenate(list(rneg.values())), np.concatenate(list(epos.values())))
rng = np.random.default_rng(0)
draws = {m: [] for m in mons}
for _ in range(2000):
    ir = np.concatenate([rneg[c] for c in rng.choice(rc, len(rc))])
    ip = np.concatenate([epos[c] for c in rng.choice(ec, len(ec))])
    p = pooled(ir, ip)
    for m in mons:
        draws[m].append(p[m])
D = {m: np.array(v) for m, v in draws.items()}
res = {"seeds": seeds, "point": point,
       "ci95": {m: [float(np.quantile(D[m], 0.025)), float(np.quantile(D[m], 0.975))] for m in mons},
       "diff_max14_minus_mean8": [float(point["max14"] - point["mean8"]),
                                  float(np.quantile(D["max14"] - D["mean8"], 0.025)), float(np.quantile(D["max14"] - D["mean8"], 0.975))],
       "diff_max14_minus_last14": [float(point["max14"] - point["last14"]),
                                   float(np.quantile(D["max14"] - D["last14"], 0.025)), float(np.quantile(D["max14"] - D["last14"], 0.975))]}
for m in mons:
    print(f"{m:7s} pooled catch {point[m]:.4f}  95% [{res['ci95'][m][0]:.4f}, {res['ci95'][m][1]:.4f}]")
print("max14 - mean8 ", res["diff_max14_minus_mean8"])
print("max14 - last14", res["diff_max14_minus_last14"])
(HERE / "vera_boot.json").write_text(json.dumps(res, indent=1))
