"""Vera: rebuild the behaviour gap D and its 95% interval from the saved per-prompt values (own bootstrap).

D_i = gap_i(t) - gap_i(0); D = mean D_i; CI = percentile bootstrap over the 64 prompts (own rng seed 7, 5,000
resamples), plus a check that the per-prompt means equal the parent dev_curve D.
"""
import json
import os

import numpy as np

B = os.path.expanduser("~/work/theory-1113-blind-cue-recovery/kran-research/projects/blind-cue-recovery/data/behaviour_ci.json")
P = os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit/data/runs/")
OUT = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(B))
rng = np.random.default_rng(7)
res = {}
for m, row in d.items():
    dev = {e["step"]: e["D"] for e in json.load(open(P + row["run"] + ".json"))["dev_curve"]}
    g0 = np.array(row["reads"]["0"].get("gap_i"))
    for t, r in row["reads"].items():
        g = np.array(r.get("gap_i"))
        Di = g - g0
        assert np.allclose(Di, np.array(r.get("D_i")), atol=1e-12)
        bs = Di[rng.integers(0, len(Di), (5000, len(Di)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[f"{m}|{t}"] = {"D": float(Di.mean()), "ci": [float(lo), float(hi)], "dev_curve_D": dev[int(t)], "n": len(Di)}
        print(f"{m} t={t:>3} D {100*Di.mean():5.2f} [{100*lo:5.2f}, {100*hi:5.2f}] dev_curve {100*dev[int(t)]:5.2f} "
              f"builder [{100*r['ci95'][0]:5.2f}, {100*r['ci95'][1]:5.2f}] n={len(Di)}")
json.dump(res, open(f"{OUT}/vera_beh.json", "w"), indent=1)
