#!/usr/bin/env python3
"""Recall check 1: 150 random snippets the classifier labelled ask=false (seed 11).

Writes neg_sample.jsonl (the sampled snippets with context) for hand judgement.
"""
import gzip, json, random
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

labels = {r["sid"]: r for r in map(json.loads, open(ROOT / "data" / "labels.jsonl"))}
snips = {r["sid"]: r for r in map(json.loads, gzip.open(ROOT / "data" / "snippets.jsonl.gz", "rt"))}
neg = sorted(s for s, r in labels.items() if r["ask"] is False)
print("labels", len(labels), "ask=false", len(neg), "ask=true", sum(r["ask"] is True for r in labels.values()))
rng = random.Random(11)
pick = rng.sample(neg, 150)
with open(HERE / "neg_sample.jsonl", "w") as f:
    for i, sid in enumerate(pick):
        s = snips[sid]
        f.write(json.dumps({"n": i, "sid": sid, "forum": s["forum"], "kind": s["kind"], "year": s["year"],
                            "title": s.get("post_title"), "pattern": s["pattern"], "url": s["url"],
                            "refused": bool(labels[sid].get("refused")), "snippet": s["snippet"]}) + "\n")
