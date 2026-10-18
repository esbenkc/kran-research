"""Second blind batch: 15 ensemble + 15 single-probe (lam1) round-8 answers PER SEED (60 total) NOT used in batch 1,
shuffled together with numeric ids 100-159, so the labeller cannot tell the arm. Same rubric as batch 1.
Purpose: does the judge-vs-label gap differ between the ensemble arm and the single arm (bias in the ratio)?"""
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

PG = Path("/Users/esben/work/oracle-1014-probe-goodhart/kran-research/projects/probe-goodhart")
sys.path.insert(0, str(PG))
import scenarios  # noqa: E402

PE = Path("/Users/esben/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/raw")
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]


def pid(r):
    return hashlib.sha256((r["system_message"] + r["user_query"]["content"]).encode()).hexdigest()[:12]


byid = {pid(r): r for r in scenarios._all()}
seen = {json.loads(l)["uid"] for l in open("key.jsonl")}


def load(pattern):
    xs = []
    for f in sorted(glob.glob(str(PE / pattern))):
        xs += [json.loads(l) for l in open(f)]
    return xs


ens = [x for x in load("eval_ens*_s*.jsonl") if x["round"] == 8]
single = [x for x in load("eval_lam1_s*.jsonl") if x["round"] == 8]
assert ens, "no ensemble round-8 rows yet"
rng = random.Random(10182)
out, key = [], []
for name, xs in (("ens_r8", ens), ("lam1_r8b", single)):
    xs = [x for x in xs if f"lam1_r8:{x['run']}:{x['prompt_id']}:{x['sample']}" not in seen]
    for seed in SEEDS:  # 15 per seed per arm (Felix #2353)
      xs_s = [x for x in xs if x["run"].endswith(f"_s{seed}")]
      for x in rng.sample(xs_s, 15):
        r = byid[x["prompt_id"]]
        uid = f"{name}:{x['run']}:{x['prompt_id']}:{x['sample']}"
        out.append({"uid": uid, "fact": r["context"]["sensitive_info"], "query": r["user_query"]["content"],
                    "response": x["response"]})
        key.append({"uid": uid, "judge": x["judge"], "stratum": name, "seed": seed})
rng.shuffle(out)
base = 100 + 100 * SEEDS[0] if len(SEEDS) == 1 else 100
order = {o["uid"]: base + i for i, o in enumerate(out)}
tag = "_s" + "".join(map(str, SEEDS))
with open(f"blind2{tag}.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(dict(o, n=order[o["uid"]])) + "\n")
with open(f"key2{tag}.jsonl", "w") as f:
    for k in key:
        f.write(json.dumps(dict(k, n=order[k["uid"]])) + "\n")
print(len(out), "rows; ens rows available:", len(ens))
