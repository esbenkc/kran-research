"""Vera: refit vs frozen attention-probe AUROC from data/results/refit_scores_<run>.jsonl (OTHER dropped),
with a cross-check of labels and frozen scores against the eval rows used everywhere else."""
import json, numpy as np
from vera_stats import auroc
P = "/Users/esben/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/"
for run in ["round0_base", "lam1_s0", "lam1_s1", "ens_lam1_s0", "ens_lam1_s1"]:
    rows = [r for r in map(json.loads, open(P + f"results/refit_scores_{run}.jsonl")) if r.get("judge") != "OTHER"]
    y = np.array([int(r["judge_deceptive"]) for r in rows])
    src, rd = ("lam1_s0", 0) if run == "round0_base" else (run, 8)
    ev = {(r["prompt_id"], r["sample"]): r for r in map(json.loads, open(P + f"raw/eval_{src}.jsonl")) if r["round"] == rd}
    mism = sum(int(ev[(r["prompt_id"], r["sample"])]["judge_deceptive"]) != int(r["judge_deceptive"])
               or abs(ev[(r["prompt_id"], r["sample"])]["pAttn"] - r["pAttn_frozen"]) > 1e-4 for r in rows)
    print(f"{run:<12} n={len(y)} refit {auroc([r['pAttn_refit'] for r in rows], y):.4f} "
          f"frozen {auroc([r['pAttn_frozen'] for r in rows], y):.4f}  mismatches vs eval rows: {mism}")
