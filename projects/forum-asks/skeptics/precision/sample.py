"""Precision skeptic: draw the items to judge by hand.

1. 100 random snippets with ask=true and ai=true from data/labels.jsonl (seed 7).
2. Every member of the top 10 groups in data/top50.json.
Writes sample_random.jsonl and sample_top10.jsonl (full snippet text + classifier labels) next to this file.
"""
import gzip
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

snips = {}
with gzip.open(ROOT / "data" / "snippets.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line)
        snips[r["sid"]] = r
labels = {}
with open(ROOT / "data" / "labels.jsonl") as f:
    for line in f:
        r = json.loads(line)
        labels[r["sid"]] = r

pool = sorted(sid for sid, l in labels.items() if l["ask"] and l["ai"])
print("ask & ai pool:", len(pool))
rng = random.Random(7)
picked = rng.sample(pool, 100)


def row(sid, extra=None):
    s, l = snips[sid], labels[sid]
    out = {"sid": sid, "forum": s["forum"], "kind": s["kind"], "year": s["year"], "author": s["author"],
           "url": s["url"], "title": s.get("post_title"), "pattern": s["pattern"], "snippet": s["snippet"],
           "statement": l["statement"], "field": l["field"]}
    out.update(extra or {})
    return out


with open(HERE / "sample_random.jsonl", "w") as f:
    for sid in picked:
        f.write(json.dumps(row(sid), ensure_ascii=False) + "\n")

top = json.load(open(ROOT / "data" / "top50.json"))[:10]
n = 0
with open(HERE / "sample_top10.jsonl", "w") as f:
    for g in top:
        for sid in g["members"]:
            f.write(json.dumps(row(sid, {"rank": g["rank"], "group_statement": g["statement"]}),
                               ensure_ascii=False) + "\n")
            n += 1
print("top10 members:", n)
