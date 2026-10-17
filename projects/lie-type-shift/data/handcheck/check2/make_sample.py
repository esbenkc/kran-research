"""Blind sample: answers the OLD binary judge called HONEST, 20 from base r0, 20 from lam1 r8 (seeds pooled).
Writes blind.csv (no arm/round/labels) and key.json (do not open before labelling)."""
import csv, json, random
L = "../../labels/labels_claude-sonnet-4-6.jsonl"
C = "../../contexts.jsonl"
ctx = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(C)}
rows = [json.loads(l) for l in open(L)]
base = {(r["prompt_id"], r["sample"]): r for r in rows if r["round"] == 0 and r["judge"] == "HONEST"}
base = list(base.values())
late = [r for r in rows if r["round"] == 8 and r["lam"] == 1 and r["judge"] == "HONEST"]
rng = random.Random(17)
pick = [("base", r) for r in rng.sample(base, 20)] + [("lam1_r8", r) for r in rng.sample(late, 20)]
rng.shuffle(pick)
key = []
with open("blind.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "sensitive_item", "question", "reply"])
    for i, (arm, r) in enumerate(pick):
        c = ctx[r["prompt_id"]]
        w.writerow([f"k{i:02d}", c["sensitive_info"], c["query"], r["response"]])
        key.append({"id": f"k{i:02d}", "arm": arm, "run": r["run"], "prompt_id": r["prompt_id"], "sample": r["sample"],
                    "sonnet": r["label"]})
json.dump(key, open("key.json", "w"), indent=1)
print(len(base), "base old-HONEST;", len(late), "lam1 r8 old-HONEST; wrote 40")
