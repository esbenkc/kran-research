"""Vera: variant + control policies (skeptic). Absolute flattery drop in points for every build present,
and the fixed-line R^2 where dGAP_A is valid (lower 95% bound > 0.10).
usage: vera_ctrl.py <run> [<run> ...]"""
import json, os, sys
import numpy as np
from vera_share import r2_fixed

import os
PRJ = os.environ.get("VERA_PRJ", "../../kran-research/projects/footprint-overlap")
X = json.load(open("vera_x.json"))["builds"]
tag = lambda s: s.replace("#rerun", "_rerun").replace("@", "_").replace("+emu", "_emu")
px = lambda d, b: np.load(f"{d}/{tag(b)}_plain.npz")["px"].astype(np.float64)
BD = f"{PRJ}/raw/screen/base"
out = {}
for pol in sys.argv[1:]:
    PD = f"{PRJ}/raw/behav/{pol}"
    p0, b0 = px(PD, "none"), px(BD, "none")
    dA = (p0 - px(PD, "ref@L14.q_norm")) - (b0 - px(BD, "ref@L14.q_norm"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(p0), (2000, len(p0)))
    ci = np.percentile(dA[idx].mean(1), [2.5, 97.5])
    rows = {}
    for b in X:
        if os.path.exists(f"{PD}/{tag(b)}_plain.npz"):
            dB = (p0 - px(PD, b)) - (b0 - px(BD, b))
            rows[b] = {"dgap_pts": 100 * dB.mean(), "x": X[b]["overlap"], "class": X[b]["class"], "kind": X[b]["kind"]}
    st = [b for b in rows if (rows[b]["kind"] == "cand" and rows[b]["class"] == "leaky") or b == "emu"]
    r = {"X_none": p0.mean(), "dGAP_A_pts": 100 * dA.mean(), "dGAP_A_ci_pts": (100 * ci).tolist(),
         "valid": bool(ci[0] > 0.10), "n_rows": len(rows), "n_status": len(st),
         "status_dgap_pts_mean": float(np.mean([rows[b]["dgap_pts"] for b in st])) if st else None,
         "status_dgap_pts_maxabs": float(np.max([abs(rows[b]["dgap_pts"]) for b in st])) if st else None}
    if r["valid"] and len(st) >= 6:
        y = np.array([rows[b]["dgap_pts"] / r["dGAP_A_pts"] for b in st]); x = np.array([rows[b]["x"] for b in st])
        r["R2_fixed"] = r2_fixed(y, x); r["pearson_r2"] = float(np.corrcoef(x, y)[0, 1] ** 2)
    out[pol] = r
    print(pol, json.dumps(r, default=float))
json.dump(out, open("vera_ctrl.json", "w"), indent=1, default=float)
