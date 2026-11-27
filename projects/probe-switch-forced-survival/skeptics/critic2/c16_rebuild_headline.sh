# critic2 (openai/gpt-5.5) tool call 16, verbatim from its session; cwd = /Users/esben/work/oracle-1127-forced-switch-head
python3 -c 'import json,pathlib,statistics; base=pathlib.Path("kran-research/projects/probe-switch-forced-survival/data/kaudit"); vals=[]; r8=[]; kmins=[]
for s in range(3):
 d=json.load(open(base/("k6_s%d.json"%s))); h=d["headline"]; sk=h["strict32_after_greedy_k"]["TB"]; km=h["strict32_smallest_k_below_0.25"]["TB"]; rr=h["strict32_random_min_median_max"]["8"]
 vals.append(sk[7]); r8.append(rr[1]); kmins.append(13 if km is None else km); print("k6_s%d: S0=%.3f S8=%.3f S12=%.3f k_min=%s R8med=%.3f"%(s,h["base_strict32"],sk[7],sk[11],("gt12" if km is None else km),rr[1]))
print("status %s S8 %s mean %.3f R8 %s kmins %s"%(("REJECTED" if any(v>=0.25 for v in vals) else "CONFIRMED"),vals,statistics.mean(vals),r8,kmins))'
