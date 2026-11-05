"""Reading cells that need the saved vectors (PREREG s6): error fraction |g_real - g_pred| / |g_real - g_frz| per
token (how much of the frozen-norm error the first-order tick leaves), median + bootstrap CI per model/set/layer.
-> data/vector_stats.json"""
import glob
import json
import os

import numpy as np

from analyze import boot

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "data", "runs")
out = []
for f2 in sorted(glob.glob(os.path.join(R, "*_s2.npz"))):
    base = os.path.basename(f2)[:-len("_s2.npz")]
    model, setname = base.split("_")
    if setname == "S":
        continue
    s1, s2 = np.load(os.path.join(R, base + "_s1.npz")), np.load(f2)
    for k in s2.files:
        L = int(k[1:].split("_")[0])
        gr, gf, gp = s2[k].astype(np.float64), s1[f"L{L}_frz"].astype(np.float64), s1[f"L{L}_pred"].astype(np.float64)
        num = np.linalg.norm(gr - gp, axis=1)
        den = np.linalg.norm(gr - gf, axis=1)
        frac = num / den
        m, lo, hi = boot(frac)
        out.append({"model": model, "set": setname, "layer": L, "error_fraction_left_by_tick":
                    {"median": m, "ci_lo": lo, "ci_hi": hi}, "n": len(frac)})
json.dump(out, open(os.path.join(HERE, "data", "vector_stats.json"), "w"), indent=1)
for r in out:
    e = r["error_fraction_left_by_tick"]
    print(r["model"], r["set"], r["layer"], f"{e['median']:.3f} [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]")
