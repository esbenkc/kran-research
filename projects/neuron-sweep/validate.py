"""Judge vs. blind hand labels: confusion matrix, accuracy, Cohen's kappa.

    uv run python projects/neuron-sweep/validate.py data/api_hand.jsonl hand_pool_a.jsonl dev_x10.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
key = lambda r: f'{r["kind"]}|{r.get("scale", 1.0):g}|{r["layer"]}|{r["neuron"]}'
lab = json.load(open(D / "hand_labels.json"))
if (D / "dev_labels.json").exists():
    lab.update(json.load(open(D / "dev_labels.json")))
ids = {key(r): r["id"] for f in sys.argv[2:] for r in map(json.loads, open(D / f))}
c = Counter()
for j in map(json.loads, open(D / Path(sys.argv[1]).name)):
    if j["key"] in ids and ids[j["key"]] in lab:
        h = lab[ids[j["key"]]]
        c[(h, j["label"])] += 1
        if h != j["label"]:
            print(ids[j["key"]], "hand", h, "judge", j["label"], j.get("thing", ""))
n = sum(c.values())
K = ["none", "mono", "poly"]
po = sum(c[(k, k)] for k in K) / n
pe = sum(sum(c[(k, x)] for x in K) * sum(c[(x, k)] for x in K) for k in K) / n / n
print("hand \\ judge", K)
for k in K:
    print(f"{k:5s}", [c[(k, x)] for x in K])
print(f"n={n} accuracy {po:.2f} kappa {(po - pe) / (1 - pe):.2f}")
