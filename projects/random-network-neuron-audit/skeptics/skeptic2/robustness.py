"""skeptic2 robustness rows: (1) one-job rate within kurtosis bins (sparsity-matched comparison, prereg row),
(2) sequence-position bias of top-20 excerpts per arm and layer (sink/early-position check).
Run from the project dir: uv run python skeptics/skeptic2/robustness.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rebuild as R  # noqa: E402

U = {u["uid"]: u for u in map(json.loads, open(PROJ / "data" / "contexts.jsonl"))}
S = {u["uid"]: u for u in json.load(open(PROJ / "data" / "units.json"))}
one = {}
for r in map(json.loads, open(PROJ / "data" / "judgments.jsonl")):
    one.setdefault(r["uid"], []).append(R.verdict(r, U[r["uid"]])["one"])
out = {"kurtosis_quantiles_10_50_90": {}, "one_job_by_kurtosis_bin": [], "position": {}}
for a in "ABC":
    k = np.array([S[u]["kurtosis"] for u in S if S[u]["arm"] == a])
    out["kurtosis_quantiles_10_50_90"][a] = np.percentile(k, [10, 50, 90]).round(1).tolist()
for lo, hi in [(0, 10), (10, 30), (30, 100), (100, float("inf"))]:
    for a in "ABC":
        us = [u for u in S if S[u]["arm"] == a and lo <= S[u]["kurtosis"] < hi]
        if us:
            out["one_job_by_kurtosis_bin"].append({"arm": a, "bin": [lo, hi], "n": len(us),
                                                   "one_job": float(np.mean([np.mean(one[u]) for u in us]))})
for a in "ABC":
    for L in (4, 14, 24):
        pos = np.array([e["pos"] for u in U.values() if u["arm"] == a and u["layer"] == L for e in u["examples"]])
        out["position"][f"{a}/L{L}"] = {"share_pos<16": float((pos < 16).mean()), "share_pos<64": float((pos < 64).mean()),
                                        "median_pos": float(np.median(pos))}
out["position"]["uniform_share_pos<16"] = 12 / 252   # eligible positions 4..255
out["position"]["uniform_share_pos<64"] = 60 / 252
json.dump(out, open(HERE / "out" / "robustness.json", "w"), indent=1)
print(json.dumps(out, indent=1))
