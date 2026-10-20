"""Rebuild build's pooled fact-present fit numbers (data/fits/pooled_fact.json: 4000 rows = 4 trained policies x 1000,
policy order from the file) with my AUROC: spin vs fact-mentioning truthful, per policy.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_pooled.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, classes, load_labels  # noqa: E402
from vera_io import D20, P  # noqa: E402

d = json.load(open(P / "data" / "fits" / "pooled_fact.json"))
print("rows:", d["rows"], "| policies:", d["policies"])
for k, v in d["scores"].items():
    s = np.nanmean(np.array(v, float), 1)
    out = []
    for i, m in enumerate(d["policies"]):
        c = classes(load_labels(m, D20)[:1000]); x = s[1000 * i:1000 * (i + 1)]
        out.append(f"{m} {auroc(x[c['spin']], x[c['truthful_fs']]):.3f}")
    print(k, " ".join(out))
