#!/usr/bin/env python3
"""Hugo: 'perfect reader' of r_L. Projection of last-prompt-token acts onto r_L (from base FIT).
(a) oracle: threshold on TEST itself, flags at most 9/200 TEST harmless (< 5%); share of TEST harmful above it.
(b) honest: threshold = the 4th-highest DEV harmless score (flags 3/64 DEV harmless), applied to TEST.
Usage: python3 hugo_reader.py <runs_dir> [seed]"""
import sys, json
import numpy as np

R = sys.argv[1]
b = np.load(f"{R}/base/acts.npz")
seed = sys.argv[2] if len(sys.argv) > 2 else "0"
s = np.load(f"{R}/strip_s{seed}/acts.npz")
out = {}
for L in range(1, 28):
    r = b["fit_harmful"][:, L].astype(np.float64).mean(0) - b["fit_harmless"][:, L].astype(np.float64).mean(0)
    row = {}
    for name, z in (("base", b), ("stripped", s)):
        p = {k: (z[k][:, L].astype(np.float64) * r).sum(1) for k in ("dev_harmless", "test_harmful", "test_harmless")}
        tn = np.sort(p["test_harmless"])[::-1]
        t_or = tn[9]  # strictly above the 10th-highest -> at most 9 harmless flagged
        fa_or = int((p["test_harmless"] > t_or).sum())
        dn = np.sort(p["dev_harmless"])[::-1]
        t_dev = dn[3]  # strictly above the 4th-highest -> 3/64 DEV harmless flagged
        row[name] = {"oracle_catch": round(float((p["test_harmful"] > t_or).mean()), 3), "oracle_fa_n": fa_or,
                     "devthr_catch": round(float((p["test_harmful"] > t_dev).mean()), 3),
                     "devthr_test_fa": round(float((p["test_harmless"] > t_dev).mean()), 3)}
    out[L] = row
print(json.dumps(out, indent=0))
