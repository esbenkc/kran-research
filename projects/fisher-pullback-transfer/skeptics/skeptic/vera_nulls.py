"""Vera's null tests and sensitivity rows, from rebuild.json (measured drops + per-run delta_w) and clean data.

1. random-metric null: P = Q diag(eig(PF)) Q', Q random rotation (spectrum kept), 100 draws
2. within-target permutation of the Fisher prediction across probes, 1000 draws
3. damping grid c in {0.01,0.1,1,10,100} of median eig; also F-2000 and per-token F
Primary set, exact map (the verdict map).
"""
import json
import sys
from collections import defaultdict

import numpy as np

from vera_common import D, load, metrics, pred_drop_exact, r2, r_P, test_set

out_dir = sys.argv[1] if len(sys.argv) > 1 else "rebuild"
RB = json.load(open(f"{out_dir}/rebuild.json"))
L = load()
W = L["W"]
pairs = RB["primary_pairs"]
meas = np.array([p["meas"] for p in pairs])
ratios = defaultdict(list)
for c in RB["checks"]:
    ratios[c["target"]].append(c["my_delta_w"] / c["my_Gw"])
cache = {}


def predict(P):
    out = []
    for p in pairs:
        t, un = p["target"], p["probe"]
        w, u = W[L["name2i"][t]], W[L["name2i"][un]]
        if un not in cache:
            cache[un] = test_set(L, un, 1)
        Xu, yu = cache[un]
        r = r_P(P, u, w)
        out.append(np.mean([pred_drop_exact(Xu, yu, u, w, q, r) for q in ratios[t]]))
    return np.array(out)


F8 = np.load(D / "fisher/fisher_big.npz")["F_common"]
Ms, lam = metrics(L, 1.0, F8)
base = predict(Ms["fisher"])
R2F = r2(base, meas)
print(f"R2_F (F-8000, c=1) = {R2F:.4f}; check vs rebuild: {RB['summary']['primary']['R2']['fisher']:.4f}")
res = {"R2_F": R2F}

# 3. damping + F variants
fz = L["fz"]
for name, F in [("F8000", F8), ("F2000", fz["F_common"]), ("Ftoken2000", fz["F_token"])]:
    for c in [0.01, 0.1, 1, 10, 100]:
        M, l = metrics(L, c, F)
        v = r2(predict(M["fisher"]), meas)
        res[f"{name}_c{c}"] = v
        print(f"   {name} c={c:<5} lambda={l:.5f}  R2_F={v:.4f}")

# 1. random rotation null
rng = np.random.default_rng(1)
ev = np.linalg.eigvalsh(Ms["fisher"])
null = []
for i in range(100):
    Q, _ = np.linalg.qr(rng.normal(size=(512, 512)))
    null.append(r2(predict(Q @ np.diag(ev) @ Q.T), meas))
null = np.array(null)
print(f"random-rotation metric null: mean R2 {null.mean():.4f}, 95th pct {np.percentile(null,95):.4f}, "
      f"max {null.max():.4f}; frac >= real {np.mean(null >= R2F):.3f}")
res["rot_null"] = dict(mean=null.mean(), p95=np.percentile(null, 95), frac_ge=float(np.mean(null >= R2F)))

# 2. within-target permutation
tg = np.array([p["target"] for p in pairs])
perm = []
for i in range(1000):
    b = base.copy()
    for t in set(tg):
        idx = np.where(tg == t)[0]
        b[idx] = b[rng.permutation(idx)]
    perm.append(r2(b, meas))
perm = np.array(perm)
print(f"within-target permutation null: mean R2 {perm.mean():.4f}, 95th {np.percentile(perm,95):.4f}; "
      f"frac >= real {np.mean(perm >= R2F):.3f}")
res["perm_null"] = dict(mean=perm.mean(), p95=np.percentile(perm, 95), frac_ge=float(np.mean(perm >= R2F)))
json.dump(res, open(f"{out_dir}/nulls.json", "w"), indent=1, default=float)
