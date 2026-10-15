"""Summarize N3: detection hits with a mismatched label (another unit's label, same arm+layer+label judge)
vs the matched label from data/judgments.jsonl for the same units and label judge."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
N3 = [json.loads(l) for l in open(HERE / "out" / "n3_mismatch_detection.jsonl")]
matched = {(r["uid"], r["judge"]): r["hits"] for r in map(json.loads, open(PROJ / "data" / "judgments.jsonl"))}
res = defaultdict(dict)
for arm in sorted({r["arm"] for r in N3}):
    rs = [r for r in N3 if r["arm"] == arm and r["hits"] is not None]
    mm = [r["hits"] for r in rs]
    mt = [matched[(r["uid"], r["label_judge"])] for r in rs if matched.get((r["uid"], r["label_judge"])) is not None]
    res[arm] = {"n": len(rs), "mismatch_mean_hits": sum(mm) / len(mm), "mismatch_share_hits>=4": sum(h >= 4 for h in mm) / len(mm),
                "matched_mean_hits": sum(mt) / len(mt), "matched_share_hits>=4": sum(h >= 4 for h in mt) / len(mt)}
res["chance_mean_hits"] = 2.5
res["chance_share_hits>=4"] = 26 / 252
json.dump(res, open(HERE / "out" / "n3_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
