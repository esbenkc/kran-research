"""E2/E2b summary (EXPLORATORY): the 8B LLC estimate of k2 seed 5 at updates 32 and 88 versus the step constant c
(frozen budget), and the running estimate of the long chain. Recomputed from the per-draw losses, not from lambda_hat.

    .venv/bin/python e2_summary.py      # -> data/llc/e2_summary.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
out = {"frozen_budget": [], "long": None}
for f in sorted((HERE / "data/llc/e2").glob("m8_*.json")):
    x = json.load(open(f))
    draws = [l[x["burn"]:] for l in x["losses"]]
    lam = float(np.mean([x["nbeta"] * np.mean(d) for d in draws]))
    out["frozen_budget"].append({"step": x["step"], "c": x["c"], "eps": x["eps"], "lambda_hat": lam,
                                 "chains": [x["nbeta"] * float(np.mean(d)) for d in draws], "bm_se": x["bm_se"],
                                 "L_full_last_draw": x.get("L_full_last_draw"), "file": str(f.relative_to(HERE))})
rows = out["frozen_budget"]
cs = sorted({r["c"] for r in rows})
table = []
for c in cs:
    a = next((r for r in rows if r["c"] == c and r["step"] == 32), None)
    b = next((r for r in rows if r["c"] == c and r["step"] == 88), None)
    if a and b:
        table.append({"c": c, "u32": a["lambda_hat"], "u88": b["lambda_hat"], "diff": b["lambda_hat"] - a["lambda_hat"],
                      "ratio": b["lambda_hat"] / a["lambda_hat"],
                      "diff_se": (a["bm_se"] ** 2 + b["bm_se"] ** 2) ** 0.5})
out["table"] = table
if len(table) >= 2:
    lc = np.log([t["c"] for t in table])
    for k in ("u32", "u88"):
        out[f"loglog_slope_{k}"] = float(np.polyfit(lc, np.log([t[k] for t in table]), 1)[0])
long = sorted((HERE / "data/llc/e2b").glob("m8_*.json"))
if long:
    x = json.load(open(long[0]))
    run = []
    for k in range(100, len(x["losses"][0]) + 1, 100):  # running estimate: mean of steps (k/4, k] over both chains
        lo = k // 4
        run.append({"steps": k, "lambda_hat": float(np.mean([x["nbeta"] * np.mean(l[lo:k]) for l in x["losses"]]))})
    out["long"] = {"file": str(long[0].relative_to(HERE)), "c": x["c"], "step": x["step"], "burn": x["burn"],
                   "draws": x["draws"], "lambda_hat": x["lambda_hat"], "bm_se": x["bm_se"], "running": run,
                   "rule": "running estimate at k steps = n beta x mean minibatch loss over steps k/4+1..k, both chains"}
(HERE / "data/llc/e2_summary.json").write_text(json.dumps(out, indent=1))
for t in table:
    print(f"c {t['c']}: u32 {t['u32']:.3f}  u88 {t['u88']:.3f}  diff {t['diff']:+.3f} (SE {t['diff_se']:.3f})  "
          f"ratio {t['ratio']:.3f}")
for k in ("loglog_slope_u32", "loglog_slope_u88"):
    if k in out:
        print(k, round(out[k], 3))
if out["long"]:
    print("long", [(r["steps"], round(r["lambda_hat"], 3)) for r in out["long"]["running"][::4]])
