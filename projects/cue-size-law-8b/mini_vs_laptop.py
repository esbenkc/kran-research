"""Determinism check (Bruno, theory-1118): every mini run m8_* against its laptop copy l8_* (labq J1/J2).

Compares the saved eval curves point by point (dense_curve + dev_curve, every eval step), not only T.
Same cache (sha256 in data/laptop/J2/cache_check.json), same code, same seed: equal curves mean the laptop
seeds 3-4 come from the same pipeline as the mini seeds 0-2. It is NOT an independent replication.

    python mini_vs_laptop.py   # writes data/mini_vs_laptop.json
"""
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).parent


def curve(run):
    pts = {}
    for p in (run.get("dense_curve") or []) + run.get("dev_curve", []):
        pts[int(p["step"])] = float(p["D"])
    return pts


rows = []
for m in sorted(glob.glob(str(HERE / "data" / "runs" / "m8_*.json"))):
    lap = m.replace("/m8_", "/l8_")
    if not os.path.exists(lap):
        continue
    M, L = json.load(open(m)), json.load(open(lap))
    cm, cl = curve(M), curve(L)
    common = sorted(set(cm) & set(cl))
    diffs = [abs(cm[s] - cl[s]) for s in common]
    rows.append(dict(
        cell=os.path.basename(m)[3:-5], machines=[M.get("machine"), L.get("machine")],
        n_mini=len(cm), n_laptop=len(cl), n_common=len(common),
        n_equal=sum(d == 0.0 for d in diffs), max_abs_diff=max(diffs) if diffs else None,
        T_dense=[M.get("T_dense"), L.get("T_dense")], resumed_from=[M.get("resumed_from"), L.get("resumed_from")],
        max_D=[max(cm.values()), max(cl.values())],
        test_gap=[(M.get("test") or {}).get("gap"), (L.get("test") or {}).get("gap")],
    ))

summary = dict(
    cells=len(rows),
    cells_all_equal=sum(r["n_equal"] == r["n_common"] == r["n_mini"] == r["n_laptop"] for r in rows),
    points=sum(r["n_common"] for r in rows), points_equal=sum(r["n_equal"] for r in rows),
    max_abs_diff=max(r["max_abs_diff"] for r in rows),
)
cache = HERE / "data" / "laptop" / "J2" / "cache_check.json"
if cache.exists():
    summary["cache_check_J2"] = json.load(open(cache))
json.dump(dict(summary=summary, rows=rows), open(HERE / "data" / "mini_vs_laptop.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:1500])
for r in rows:
    print(r["cell"], r["n_common"], r["n_equal"], r["max_abs_diff"], r["T_dense"], r["resumed_from"])
