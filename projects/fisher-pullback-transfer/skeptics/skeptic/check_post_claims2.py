"""Check 'repeat runs agree at r=0.98, 0.83-0.87 below 0.02' and 'spread widened 1.6-6x' from rebuild + run jsons."""
import glob, json, numpy as np
from collections import defaultdict
from vera_common import D
RB = json.load(open("rebuild/rebuild.json"))
# per-seed measured drops: recompute from run jsons' stored AUROCs (matched mine to 5e-5)
scored = json.load(open(D / "selection.json"))["scored_probes"]
by = defaultdict(dict); sd_ratio = []
for jf in sorted(glob.glob(str(D / "runs/*__seed*.json"))):
    if "lam0" in jf: continue
    j = json.load(open(jf))
    if not j["success"]: continue
    for r in j["probes"]:
        if r.get("set", "primary") == "primary" and r["probe"] in scored and r["probe"] != j["target"]:
            by[j["seed"]][(j["target"], r["probe"])] = r["auroc_clean"] - r["auroc_attacked"]
        if r["probe"] == j["target"]:
            sd_ratio.append((j["target"], j["seed"], r["sd_pooled_attacked"] / r["sd_pooled_clean"]))
for a, b in [(0, 1), (0, 2), (1, 2)]:
    ks = sorted(set(by[a]) & set(by[b])); x = np.array([by[a][k] for k in ks]); y = np.array([by[b][k] for k in ks])
    small = (np.abs(x) < 0.02) & (np.abs(y) < 0.02)
    print(f"seeds {a}-{b}: n={len(ks)} r={np.corrcoef(x,y)[0,1]:.3f}; both |drop|<0.02: n={small.sum()} r={np.corrcoef(x[small],y[small])[0,1]:.3f}")
print("target sd ratio attacked/clean:", [(t, s, round(v, 2)) for t, s, v in sd_ratio])
# alternative filter: seed-mean |drop| < 0.02 (rebuild pairs)
mean = {(p["target"], p["probe"]): p["meas"] for p in RB["primary_pairs"]}
for a, b in [(0, 1), (0, 2), (1, 2)]:
    ks = [k for k in sorted(set(by[a]) & set(by[b])) if abs(mean[k]) < 0.02]
    x = np.array([by[a][k] for k in ks]); y = np.array([by[b][k] for k in ks])
    print(f"seeds {a}-{b}: seed-mean |drop|<0.02 filter: n={len(ks)} r={np.corrcoef(x,y)[0,1]:.3f}")
