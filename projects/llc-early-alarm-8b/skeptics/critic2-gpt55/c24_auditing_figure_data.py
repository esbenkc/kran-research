# critic2 (openai/gpt-5.5) tool call 24, "Auditing figure data", verbatim from its session; run from /Users/esben/work/theory-1124-llc-early-alarm-8b

import json, statistics, math
from pathlib import Path
base=Path("kran-research/projects/llc-early-alarm-8b"); fig=json.load(open("blog.kran.ai/assets/figures/llc-early-alarm-8b/data.json")); bud=json.load(open(base/"data/gate/budget.json"))["rows"]
maxdiff=0; mism=0
for line in fig["lines"]:
  hs=[r["lambda_hat"] for r in bud if r["r"]==line["rank"] and abs(r["c"]-line["c"])<1e-12 and abs(r["gamma"]-line["gamma"])<1e-12 and r["steps"]==line["steps"]]
  if hs:
    diff=abs(statistics.mean(hs)/line["truth"]-line["y"]); maxdiff=max(maxdiff,diff)
  else: mism += 1
print("figure_lines_checked",len(fig["lines"])-mism,"missing",mism,"max_y_diff",maxdiff)
# Check first 48 walk rows match raw distances/noise ratios
dim=2424832; walk_rows=[]
for u in [32,88]:
 for cstr in ["0.033","0.1","0.3"]:
  d=json.load(open(base/f"data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u{u:05d}_c{cstr}.json")); c=float(cstr)
  for ci,ch in enumerate(d["dist_from_wstar"]):
   for t,dist in zip([25,50,75,100], ch): walk_rows.append((c,u,ci,t,dist,math.sqrt(d["eps"]*t*dim)))
print("raw_walk_rows", len(walk_rows), "fig_walk_rows", len(fig["walks"]))
md=0; mn=0
for got, exp in zip(fig["walks"], walk_rows):
 c,u,ci,t,dist,noise=exp
 md=max(md, abs(got["dist"]-dist)); mn=max(mn, abs(got["noise"]-noise))
print("walk_max_dist_diff", md, "walk_max_noise_diff", mn, "fig_ratio_range", fig["ratio_range"])

