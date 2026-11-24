# critic2 (openai/gpt-5.5) tool call 22, "Checking running means", verbatim from its session; run from /Users/esben/work/theory-1124-llc-early-alarm-8b
import json, statistics
d=json.load(open("kran-research/projects/llc-early-alarm-8b/data/llc/e2b/m8_k2_s5_cue-m8_k2_s5_u00088_c0.1_long.json"))
nb=d["nbeta"]; base=d["L_wstar_all_rows"]
for t in [100,500,900,1600,2500,3200]:
    b=t//4; vals=[]
    for losses in d["losses"]: vals += losses[b:t]
    print(t, round(nb*(statistics.mean(vals)-base),1))

