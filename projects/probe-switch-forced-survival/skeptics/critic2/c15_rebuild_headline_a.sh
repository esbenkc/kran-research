# critic2 (openai/gpt-5.5) tool call 15, verbatim from its session; cwd = /Users/esben/work/oracle-1127-forced-switch-head
python3 -c 'import json, pathlib, statistics; base=pathlib.Path("kran-research/projects/probe-switch-forced-survival/data/kaudit");
vals=[]; r8=[]; kmins=[]
for s in range(3):
 d=json.load(open(base/f"k6_s{s}.json")); h=d["headline"]; sk=h["strict32_after_greedy_k"]["TB"]; km=h["strict32_smallest_k_below_0.25"]["TB"]; rr=h["strict32_random_min_median_max"]["8"]
 vals.append(sk[7]); r8.append(rr[1]); kmins.append(13 if km is None else km); print(f"k6_s{s}: S0={h[\"base_strict32\"]:.3f} S8={sk[7]:.3f} S12={sk[11]:.3f} k_min={km or \"gt12\"} R8med={rr[1]:.3f}")
print("status", "REJECTED" if any(v>=0.25 for v in vals) else "CONFIRMED", "S8", vals, "mean", statistics.mean(vals), "R8", r8, "kmins", kmins)
'
