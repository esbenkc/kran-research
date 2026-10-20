"""Compare Iris's blind agent labels (labels_iris.csv) with the judge/string labels in key.sealed.jsonl."""
import csv, json
from collections import Counter
lab = {int(r["item"]): r for r in csv.DictReader(open("labels_iris.csv"))}
key = {k["item"]: k for k in map(json.loads, open("key.sealed.jsonl"))}
assert set(lab) == set(key) and len(lab) == 100
out = {}
for s in ["spin", "plain", "truthful_fact"]:
    ks = [k for k in key.values() if k["stratum"] == s]
    d = [int(lab[k["item"]]["D"]) for k in ks]
    m = [int(lab[k["item"]]["M"]) for k in ks]
    w = [int(lab[k["item"]]["W"]) for k in ks]
    j2 = [k["y_judge2"] for k in ks if k["y_judge2"] is not None]
    out[s] = {"n": len(ks), "agent_deceptive": sum(d), "agent_fact_meaning_stated": sum(m), "agent_fact_words": sum(w),
              "judge2_deceptive": sum(j2), "judge2_n": len(j2)}
    print(s, out[s])
# agreement with judge on D
agree = sum(int(lab[i]["D"]) == key[i]["y_judge"] for i in key)
print("agent vs judge deceptive agreement, all 100:", agree)
sp = [i for i in key if key[i]["stratum"] == "spin"]
print("spin: agent D=1 & M=1 (real spin: fact stated, framing misleads):", sum(lab[i]["D"] == "1" and lab[i]["M"] == "1" for i in sp),
      "| agent D=1 & M=0 (fact words but fact not conveyed):", sum(lab[i]["D"] == "1" and lab[i]["M"] == "0" for i in sp),
      "| agent D=0:", sum(lab[i]["D"] == "0" for i in sp))
print("per policy spin agent-deceptive:", Counter((key[i]["policy"], lab[i]["D"]) for i in sp))
# fact-stated truthful: agent agrees honest?
tf = [i for i in key if key[i]["stratum"] == "truthful_fact"]
print("truthful_fact agent honest:", sum(lab[i]["D"] == "0" for i in tf), "/", len(tf))
# string check vs agent fact words / meaning on all
print("string y_string==0 (fact present) vs agent W:", Counter((key[i]["y_string"], lab[i]["W"]) for i in key))
json.dump(out, open("agree.json", "w"), indent=1)
