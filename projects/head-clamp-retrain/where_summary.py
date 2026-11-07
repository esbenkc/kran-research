"""data/where_summary.json: per audited run (data/audit/*.json, unmerged adapter, run's clamp on): the head whose
single swap cut removes the most of the returned push (dev), and test strict / push before and after that cut,
recomputed from the per-prompt rows; plus the update split (attention-only / MLP-only) test strict, and the D5
P-COND check (top head in the frozen conditional t=0 top 3 of Vera, DEVIATIONS.md D5).

    $PY where_summary.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PCOND = {1: ["27.12", "27.6", "27.15"], 2: ["27.7", "27.6", "27.3"]}  # DEVIATIONS.md D5 (frozen 06:36-06:45 PST)


def strict_push(rows, a_id=32, b_id=33):
    d = {}
    for p, pr, top, la, lb in rows:
        d.setdefault(p, {})[pr] = (top, la - lb)
    s = sum(1 for v in d.values() if v[1][0] == a_id and v[0][0] == b_id)
    return {"strict": s / len(d), "n": len(d), "push": sum(v[1][1] - v[0][1] for v in d.values()) / len(d)}


out = {}
for f in sorted((HERE / "data" / "audit").glob("*.json")):
    a = json.load(open(f))
    top, share = a["rank_dev"][0]
    r = {"seed": a["seed"], "clamped": a["clamped"], "top_head": top, "top_removed_dev": share,
         "top_removed_test": a["single_cut_test"][top], "second": a["rank_dev"][1] if len(a["rank_dev"]) > 1 else None,
         "before_test": strict_push(a["rows"]["run/test"]), "after_top_cut_test": strict_push(a["rows"][f"cut{top}/test"]),
         "split_test_strict": {k: v["test"]["switch_correct"] for k, v in a["split"].items()},
         "split_test_push": {k: v["test"]["shift_mean"] for k, v in a["split"].items()}}
    if a["run"].startswith("set_") and a["seed"] in PCOND:
        r["pcond_top3"] = PCOND[a["seed"]]
        r["pcond_hit"] = None if share < 0.30 else top in PCOND[a["seed"]]
    out[a["run"]] = r
(HERE / "data" / "where_summary.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v["top_head"], round(v["top_removed_dev"], 3), "test", v["before_test"]["strict"], "->",
          v["after_top_cut_test"]["strict"], "pcond", v.get("pcond_hit"), "split", v["split_test_strict"])
