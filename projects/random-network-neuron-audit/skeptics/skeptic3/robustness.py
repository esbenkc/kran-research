"""skeptic3 robustness rows from raw judgments (own code).

1. Fit-threshold sweep: one-job rate per arm when the '>= 15 of 20 fit' cut moves to 8/10/12/15/18
   (detection gate >= 4/5 kept). Shows which quantity is robust (the arm ordering) and which is not (the level).
2. Detection hits per arm vs the hypergeometric chance mean of 2.5 (5 picks, 5 true among 10).
3. Specificity sanity: units with judge-free lexical purity >= 0.75 (one token dominates) and their verdicts.
Writes out/robustness.json.   Run: uv run python skeptics/skeptic3/robustness.py  (from the project dir)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from rebuild import purity, verdicts

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
rows = [json.loads(l) for l in open(P / "data" / "judgments.jsonl")]
units = {u["uid"]: u for u in map(json.loads, open(P / "data" / "contexts.jsonl"))}
res: dict = {"threshold_sweep": {}, "detection": {}, "pure_lexical_units": []}
for arm in "ABC":
    rs = [r for r in rows if r["arm"] == arm]
    sweep = {}
    for t in (8, 10, 12, 15, 18):
        ok = [(r["parsed"] or {}).get("main_type") not in (None, "NONE") and (r["parsed"] or {}).get("main_count", 0) >= t
              and (r.get("hits") or 0) >= 4 for r in rs]
        sweep[t] = float(np.mean(ok))
    res["threshold_sweep"][arm] = sweep
    h = [r["hits"] for r in rs if r["hits"] is not None]
    res["detection"][arm] = dict(n=len(h), mean_hits=float(np.mean(h)), chance_mean=2.5,
                                 share_ge4=float(np.mean([x >= 4 for x in h])), dist=dict(sorted(Counter(h).items())))
v = verdicts(rows, "pipeline")
for uid, u in units.items():
    p = purity(u["examples"])
    if p >= 0.75:
        res["pure_lexical_units"].append(dict(arm=u["arm"], layer=u["layer"], purity=p,
                                              one=[v[uid][j]["one"] for j in sorted(v[uid])],
                                              labels=[v[uid][j]["label"] for j in sorted(v[uid])]))
json.dump(res, open(HERE / "out" / "robustness.json", "w"), indent=1)
print(json.dumps(res, indent=1))
