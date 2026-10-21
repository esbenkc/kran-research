#!/usr/bin/env python3
"""Hugo: harmful/harmless mean position along r_L (units of |r_L|: h.r/|r|^2), base vs stripped, from acts.npz.
r_L rebuilt from BASE FIT acts (not dirs.npy). Usage: python3 hugo_gap.py <runs_dir> [layer]"""
import sys, json
import numpy as np

R = sys.argv[1]
Ls = [int(sys.argv[2])] if len(sys.argv) > 2 else list(range(1, 28))
b = np.load(f"{R}/base/acts.npz")
s = np.load(f"{R}/strip_s0/acts.npz")
r = b["fit_harmful"].astype(np.float64).mean(0) - b["fit_harmless"].astype(np.float64).mean(0)
out = {}
for L in Ls:
    rv = r[L]
    nn = rv @ rv
    row = {}
    for name, z in (("base", b), ("stripped", s)):
        for split in ("dev", "test"):
            h = z[f"{split}_harmful"][:, L, :].astype(np.float64) @ rv / nn
            n = z[f"{split}_harmless"][:, L, :].astype(np.float64) @ rv / nn
            row[f"{name}_{split}"] = [round(h.mean(), 3), round(n.mean(), 3), round(h.mean() - n.mean(), 3)]
    out[L] = row
print(json.dumps(out, indent=0))
