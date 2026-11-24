# critic2 (openai/gpt-5.5) tool call 17, "Rebuilding exploratory metrics", verbatim from its session; run from /Users/esben/work/theory-1124-llc-early-alarm-8b

import json, math, statistics
from pathlib import Path
b=Path("kran-research/projects/llc-early-alarm-8b")
bud=json.load(open(b/"data/gate/budget.json"))["rows"]
for S in [100,400,1600,6400]:
 vals=[]
 for rank,truth in [(1,236),(8,312)]:
  hs=[x["lambda_hat"] for x in bud if x.get("r")==rank and abs(x.get("c")-0.1)<1e-12 and abs(x.get("gamma")-1)<1e-12 and x.get("steps")==S]
  vals.append((rank, statistics.mean(hs), (statistics.mean(hs)-truth)/truth, len(hs), [round(h,1) for h in hs]))
 print("budget", S, [(r, round(h,1), round(e,3), n, hs) for r,h,e,n,hs in vals])
for u in [32,88]:
 for cstr in ["0.033","0.1","0.3"]:
  p=b/f"data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u{u:05d}_c{cstr}.json"
  d=json.load(open(p)); ratios=[]
  for ch in d["dist_from_wstar"]:
    for item in ch:
      if item["t"] in (25,50,75,100): ratios.append(item["ratio_to_noise"])
  print("e2",u,cstr,"lambda",round(d["lambda_hat"],3),"chains",[round(x,2) for x in d["chains"]],"se",round(d["bm_se"],3),"ratio",round(min(ratios),4),round(max(ratios),4),"eps",format(d["eps"],".3g"))
d=json.load(open(b/"data/llc/e2b/m8_k2_s5_cue-m8_k2_s5_u00088_c0.1_long.json")); print("e2b", round(d["lambda_hat"],3), [round(x,3) for x in d["chains"]], round(d["bm_se"],3), "secs", d["secs"], "loss_lens", [len(x) for x in d["losses"]])
for t in [100,500,900,1600,2500,3200]:
 vals=[]
 for losses in d["losses"]:
  vals += losses[:t]
 print("runmean",t, round(d["nbeta"]*(statistics.mean(vals)-d["L_wstar_all_rows"]),1))
for a,b2 in [(800,2000),(2000,3200),(2400,3200),(1600,2400)]:
 vals=[]
 for losses in d["losses"]: vals += losses[a:b2]
 print("window",a+1,b2, round(d["nbeta"]*(statistics.mean(vals)-d["L_wstar_all_rows"]),2))
last=[]
for ch in d["dist_from_wstar"]:
 for item in ch:
  if item["t"]==3200: last.append(item.get("ratio_to_noise"))
print("e2b_distance3200", [round(x,4) for x in last])

