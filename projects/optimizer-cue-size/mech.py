"""Mechanism read from the per-update logs already on disk (Tove #20355): along the cue-read direction d (unit
gradient of the dev gap, recomputed at each parent-grid eval), each window stores the mean and SD over updates of
proj = <raw training gradient, d>. snr = proj_mean / proj_sd. Reported per headline run at four fractions of its own T
(window ends in (0, .25T], (.25T, .5T], (.5T, .75T], (.75T, T]) plus the total grad norm and the gap-gradient norm.

    /usr/bin/python3 mech.py      -> data/mech.json, printed table
"""
import json

import numpy as np

from analyze import get
from common import HERE, OPTS, SIZES, T_of

BINS = [(0, .25), (.25, .5), (.5, .75), (.75, 1.0)]
out = []
for o in OPTS:
    for a in SIZES:
        for s in (0, 1):
            r, n = get(o, a, s)
            if r is None:
                continue
            T = T_of(r) or r["steps_run"]
            w = [c for c in r["dev_curve"] if "proj_mean" in c]
            row = {"opt": o, "alpha": a, "seed": s, "run": n, "T": T, "x": r["cue_size"]["x"], "bins": []}
            for lo, hi in BINS:
                ws = [c for c in w if lo * T < c["step"] <= max(hi * T, 8)]
                if not ws:
                    row["bins"].append(None)
                    continue
                row["bins"].append({
                    "snr": float(np.mean([c["proj_mean"] / c["proj_sd"] for c in ws if c["proj_sd"] > 0])),
                    "proj_mean": float(np.mean([c["proj_mean"] for c in ws])),
                    "proj_sd": float(np.mean([c["proj_sd"] for c in ws])),
                    "grad_norm": float(np.mean([c["grad_norm"] for c in ws])),
                    "gapgrad_norm": float(np.mean([c["gapgrad_norm"] for c in ws])),
                    "normB": float(ws[-1]["normB"]), "n_windows": len(ws)})
            out.append(row)
(HERE / "data" / "mech.json").write_text(json.dumps(out, indent=1))
fmt = lambda b, k: "   -   " if b is None else f"{b[k]:7.3f}"  # noqa: E731
print("opt    x     s     T | snr by quarter of T              | grad norm by quarter        | proj_mean by quarter")
for r in out:
    print(f"{r['opt']:5} {r['x']:5.1f} {r['seed']} {r['T']:6} |" + " ".join(fmt(b, 'snr') for b in r["bins"]) + " |"
          + " ".join(fmt(b, 'grad_norm') for b in r["bins"]) + " |" + " ".join(fmt(b, 'proj_mean') for b in r["bins"]))
