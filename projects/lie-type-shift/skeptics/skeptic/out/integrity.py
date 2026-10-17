"""Skeptic integrity checks: label files vs raw rollouts; base identity; agreement incl. my 3rd judge."""
import json, collections, glob, sys
sys.path.insert(0, "..")
from paths import LABELS, ROLL
L = str(LABELS) + "/"
raw = {}
for f in glob.glob(str(ROLL) + "/eval_*.jsonl"):
    for l in open(f):
        r = json.loads(l); raw[(r["run"], r["round"], r["prompt_id"], r["sample"])] = r
base = collections.defaultdict(set)
for k, r in raw.items():
    if k[1] == 0: base[(k[2], k[3])].add(r["response"])
print("raw rows", len(raw), "| base identical across runs:", all(len(v) == 1 for v in base.values()), len(base))
J = {}
for j in ["gpt-4.1-mini", "claude-sonnet-4-6"]:
    rows = [json.loads(l) for l in open(L + f"labels_{j}.jsonl")]
    keys = [(r["run"], r["round"], r["prompt_id"], r["sample"]) for r in rows]
    mism = sum(raw[k]["response"] != r["response"] for k, r in zip(keys, rows))
    miss = set(raw) - set(keys)
    print(j, "rows", len(rows), "dupes", len(keys) - len(set(keys)), "missing", len(miss), "text mismatch", mism,
          "labels", collections.Counter(r["label"] for r in rows))
    J[j] = {k: r["label"] for k, r in zip(keys, rows)}
mine = {}
for l in open("mylabels.jsonl"):
    r = json.loads(l)
    if r["kind"] == "rollout": mine[(r["run"], r["round"], r["prompt_id"], r["sample"])] = r["label"]
J["haiku(mine)"] = mine

def kappa(a, b):
    ks = [k for k in a if k in b]; n = len(ks)
    po = sum(a[k] == b[k] for k in ks) / n
    ca, cb = collections.Counter(a[k] for k in ks), collections.Counter(b[k] for k in ks)
    pe = sum(ca[x] * cb[x] for x in ca) / n / n
    return n, round(po, 3), round((po - pe) / (1 - pe), 3)

def fb(x):  # false vs no-false vs honest/other
    return {"FALSIFICATION": "F", "OMISSION": "N", "MISLEADING_TRUE": "N"}.get(x, "H")
names = list(J)
for i in range(3):
    for k in range(i + 1, 3):
        a, b = J[names[i]], J[names[k]]
        print(names[i], "vs", names[k], "5-way n/agree/kappa", kappa(a, b),
              "| F/noF/H", kappa({x: fb(y) for x, y in a.items()}, {x: fb(y) for x, y in b.items()}))
