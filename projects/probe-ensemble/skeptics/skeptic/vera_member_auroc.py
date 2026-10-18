"""Vera: does 'fooled' mean the probes lost their ranking, or only that their fixed thresholds stopped firing?
Per-member on-policy AUROC vs judge (OTHER dropped) at r0 and r8, next to flag rates."""
import json
from vera_stats import auroc
import numpy as np
P = "/Users/esben/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/"
ens = json.load(open(P + "ensemble.json"))
for run in ["lam1_s0", "lam1_s1", "ens_lam1_s0", "ens_lam1_s1"]:
    rows = [json.loads(l) for l in open(P + f"raw/eval_{run}.jsonl")]
    for rd in (0, 8):
        R = [r for r in rows if r["round"] == rd and r["judge"] != "OTHER"]
        y = np.array([int(r["judge_deceptive"]) for r in R])
        au = {m: auroc([r[f"pE_{m}"] for r in R], y) for m in ens}
        ae = auroc([r["pEns"] for r in R], y)
        print(f"{run:<12} r{rd} ensAUROC {ae:.3f} member AUROC {min(au.values()):.3f}-{max(au.values()):.3f} "
              f"(mean {np.mean(list(au.values())):.3f})")
