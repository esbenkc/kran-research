# critic2 (openai/gpt-5.5) tool call 15, "Rebuilding headline metrics", verbatim from its session; run from /Users/esben/work/theory-1124-llc-early-alarm-8b

import json, math, statistics
from pathlib import Path
b=Path("kran-research/projects/llc-early-alarm-8b")
f=json.load(open(b/"data/gate/frozen.json")); rows=f["table"]
print("gate_n_pass", f["n_pass"], "rows", len(rows), "selected", f["selected"])
for r in rows:
    r["worst_abs_err"] = max(abs(r["r1"].get("rel_err", float("nan"))) if not math.isnan(r["r1"].get("rel_err", float("nan"))) else float("inf"), abs(r["r8"].get("rel_err", float("nan"))) if not math.isnan(r["r8"].get("rel_err", float("nan"))) else float("inf"))
finite=[r for r in rows if math.isfinite(r["worst_abs_err"])]
best=min(finite, key=lambda r:r["worst_abs_err"])
print("best_c_gamma", best["c"], best["gamma"], "r1_hat", round(best["r1"]["hat"],3), "r1_rel", round(best["r1"]["rel_err"],6), "r8_hat", round(best["r8"]["hat"],3), "r8_rel", round(best["r8"]["rel_err"],6))
worsts=[max(abs(a),abs(b)) for a,b in zip(best["r1"]["repeat_rel_errs"], best["r8"]["repeat_rel_errs"])]
mean=statistics.mean(worsts); sd=statistics.stdev(worsts); lo=mean-2.776*sd/math.sqrt(len(worsts)); hi=mean+2.776*sd/math.sqrt(len(worsts))
print("seed_worst_mean_ci", round(mean,6), round(lo,6), round(hi,6), "worsts", [round(x,6) for x in worsts])
errs=[]
for r in rows:
  for k in ("r1","r8"):
    x=r[k]
    if math.isfinite(x.get("hat", float("nan"))) and math.isfinite(x.get("bm_se", float("nan"))) and x.get("bm_se",0)>0:
      errs.append(abs(x["hat"]-x["truth"])/x["bm_se"])
print("finished_estimates", len(errs), "min_truth_SE", round(min(errs),3), "max_truth_SE", round(max(errs),1))
# budget c=.1 gamma=1 means
bud=json.load(open(b/"data/gate/budget.json"))["rows"]
for S in [100,400,1600,6400]:
 vals=[]
 for rank,truth in [(1,236),(8,312)]:
  hs=[x["lambda_hat"] for x in bud if x.get("rank")==rank and x.get("c")==0.1 and x.get("gamma")==1 and x.get("steps")==S]
  vals.append((rank, statistics.mean(hs), (statistics.mean(hs)-truth)/truth, len(hs)))
 print("budget", S, [(r, round(h,1), round(e,3), n) for r,h,e,n in vals])
# E2 short walks
for u in [32,88]:
 for c in [0.033,0.1,0.3]:
  p=b/f"data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u{u:05d}_c{c}.json"
  d=json.load(open(p)); ratios=[]
  for ch in d["dist_from_wstar"]:
    for item in ch:
      if item["t"] in (25,50,75,100): ratios.append(item["ratio_to_noise"])
  print("e2",u,c,"lambda",round(d["lambda_hat"],3),"se",round(d["bm_se"],3),"ratio",round(min(ratios),4),round(max(ratios),4))
# E2b
p=b/"data/llc/e2b/m8_k2_s5_cue-m8_k2_s5_u00088_c0.1_long.json"; d=json.load(open(p)); print("e2b_keys_loss_len", len(d["losses"][0]), len(d["losses"][1])); print("e2b", round(d["lambda_hat"],3), [round(x,3) for x in d["chains"]], round(d["bm_se"],3), "secs", d["secs"])
for t in [100,500,900,1600,2500,3200]:
 vals=[]
 for losses in d["losses"]:
  vals += losses[:t]
 print("runmean",t, round(d["nbeta"]*(statistics.mean(vals)-d["L_wstar_all_rows"]),1))
# distance ratio last
last=[]
for ch in d["dist_from_wstar"]:
 for item in ch:
  if item["t"]==3200: last.append(item.get("ratio_to_noise"))
print("e2b_distance3200", [round(x,4) for x in last])

