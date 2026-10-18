"""Draw a blind, stratified sample of eval answers for hand labelling (Iris).
Strata: round 0 (shared base), lam0 r8, lam1 r8, lam3 r8 from the probe-goodhart saved eval rows.
Output: blind.jsonl (no judge label), key.jsonl (judge label, for scoring after labelling)."""
import json, random, sys
from pathlib import Path
PG = Path("/Users/esben/work/oracle-1014-probe-goodhart/kran-research/projects/probe-goodhart")
sys.path.insert(0, str(PG))
import scenarios
# eval prompts come from common.load_splits (train.py), not scenarios.load; map by prompt_id over all rows
rows = {r["idx"]: r for r in scenarios._all()}
import hashlib
R = PG / "data/results"
def load(f): return [json.loads(l) for l in open(R / f)]
strata = {"r0": [x for x in load("eval_lam1_s0.jsonl") + load("eval_lam1_s1.jsonl") if x["round"] == 0],
          "lam0_r8": [x for x in load("eval_lam0_s0.jsonl") + load("eval_lam0_s1.jsonl") if x["round"] == 8],
          "lam1_r8": [x for x in load("eval_lam1_s0.jsonl") + load("eval_lam1_s1.jsonl") if x["round"] == 8],
          "lam3_r8": [x for x in load("eval_lam3_s0.jsonl") if x["round"] == 8]}
def pid(r): return hashlib.sha256((r["system_message"] + r["user_query"]["content"]).encode()).hexdigest()[:12]
byid = {pid(r): r for r in rows.values()}
rng = random.Random(1018)
out, key = [], []
for name, xs in strata.items():
    for x in rng.sample(xs, 15):
        r = byid[x["prompt_id"]]
        uid = f"{name}:{x['run']}:{x['prompt_id']}:{x['sample']}"
        out.append({"uid": uid, "fact": r["context"]["sensitive_info"], "query": r["user_query"]["content"],
                    "response": x["response"]})
        key.append({"uid": uid, "judge": x["judge"], "stratum": name})
rng.shuffle(out)  # mix strata so the labeller cannot tell the arm
order = {o["uid"]: i for i, o in enumerate(out)}
for o in out: o["n"] = order[o["uid"]]
key = [dict(k, n=order[k["uid"]]) for k in key]
with open("blind.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
with open("key.jsonl", "w") as f:
    for k in key: f.write(json.dumps(k) + "\n")
print(len(out), "rows")
