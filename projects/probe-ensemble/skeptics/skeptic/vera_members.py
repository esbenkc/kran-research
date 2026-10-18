"""Vera: per-member flag rates at r0/r8 from inline pE_* scores and the saved 5%-FPR thresholds."""
import json
P = "/Users/esben/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/"
ens = json.load(open(P + "ensemble.json")); attn_thr = json.load(open(P + "probe_fit_metrics.json"))["attn"]["thr_5fpr"]
for run in ["lam1_s0", "lam1_s1", "ens_lam1_s0", "ens_lam1_s1"]:
    rows = [json.loads(l) for l in open(P + f"raw/eval_{run}.jsonl")]
    for rd in (0, 8):
        R = [r for r in rows if r["round"] == rd]
        fl = {m: sum(r[f"pE_{m}"] > ens[m]["thr"] for r in R) / len(R) for m in ens}
        anyf = sum(any(r[f"pE_{m}"] > ens[m]["thr"] for m in ens) for r in R) / len(R)
        at = sum(r["pAttn"] > attn_thr for r in R) / len(R)
        print(f"{run:<12} r{rd} any {anyf:.3f} attn {at:.3f} | " + " ".join(f"{m}:{v:.2f}" for m, v in fl.items()))
