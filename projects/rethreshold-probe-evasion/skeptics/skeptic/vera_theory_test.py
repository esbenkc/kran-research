"""Vera: own computation of the PREREG theory test (E6 predicted share vs measured share at every saved check
with measured drop >= 0.2), plus per-lambda decile shift test (Tove #2752) on all runs.
Usage: uv run --with numpy --with scipy python vera_theory_test.py <runs_dir>
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_core import decomp, share_formula  # noqa: E402

ROOT = "kran-research/projects/rethreshold-probe-evasion/data/"
cz = np.load(ROOT + "clean_acts.npz")
Y, SP, S0 = cz["y"], cz["split"], cz["score"].astype(np.float64)
M = dict(refit_neg=(SP == "refit") & (Y == 0), eval_pos=(SP == "eval") & (Y == 1), eval_neg=(SP == "eval") & (Y == 0))
view = lambda s: {k: s[m] for k, m in M.items()}  # noqa: E731
C = view(S0)
sd = np.sqrt((C["eval_pos"].var() + C["eval_neg"].var()) / 2)
D = (C["eval_pos"].mean() - C["eval_neg"].mean()) / sd
print("clean D = %.3f (pooled within-class sd %.3f)" % (D, sd))
neg = (SP == "refit") | (SP == "eval")
neg &= Y == 0
q = np.quantile(S0[neg], np.linspace(0, 1, 11))
dec = np.clip(np.digitize(S0[neg], q[1:-1]), 0, 9)

per_lam = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "lam*__seed*.npz"))):
    if f.endswith("__acts.npz"):
        continue
    lam = float(re.search(r"lam([\d.]+)__", f).group(1))
    z = np.load(f)
    for i, s in enumerate(z["scores"].astype(np.float64)):
        a = view(s)
        d = decomp(C, a)
        if d["lost"] < 0.2:
            continue
        p = (C["eval_pos"].mean() - a["eval_pos"].mean()) / sd
        c = (C["eval_neg"].mean() - a["eval_neg"].mean()) / (C["eval_pos"].mean() - a["eval_pos"].mean())
        per_lam.setdefault(lam, []).append((share_formula(D, c, p), d["share"]))
    sh = S0[neg] - z["scores"][-1].astype(np.float64)[neg]
    print("%-40s decile shifts (low->high clean score): %s  top/mid ratio %.3f" % (
        os.path.basename(f), " ".join("%.2f" % sh[dec == k].mean() for k in range(10)),
        sh[dec == 9].mean() / np.mean([sh[dec == k].mean() for k in range(3, 7)])))
allv = []
for lam, v in sorted(per_lam.items()):
    v = np.array(v)
    allv.append(v)
    print("lam %-8g checks %3d  E6 MAE %.3f  mean pred %.3f  mean measured %.3f" % (
        lam, len(v), np.abs(v[:, 0] - v[:, 1]).mean(), v[:, 0].mean(), v[:, 1].mean()))
v = np.vstack(allv)
print("ALL checks %d  E6 MAE %.3f" % (len(v), np.abs(v[:, 0] - v[:, 1]).mean()))
