"""Check figure data.json against my rebuild: every point and every panel R2."""
import json, numpy as np
from vera_common import D, r2
d = json.load(open(D.parent / "figures/fisher-pullback-transfer/data.json"))
RB = json.load(open("rebuild/rebuild.json"))
print([p["label"] for p in d["panels"]], d["ticks"])
mine = {("primary", p["target"], p["probe"]): p for p in RB["primary_pairs"]}
mine.update({("secondary", p["target"], p["probe"]): p for p in RB["secondary_pairs"]})
for metric in ["plain", "whitened", "fisher"]:
    for s in ["primary", "secondary"]:
        rows = [r for r in d["rows"] if r["metric"] == metric and r["set"] == s]
        dx = max(abs(r["x"] - mine[(s, r["target"], r["probe"])][metric]) for r in rows)
        dy = max(abs(r["y"] - mine[(s, r["target"], r["probe"])]["meas"]) for r in rows)
        rr = r2([r["x"] for r in rows], [r["y"] for r in rows])
        lab = [p for p in d["panels"] if p["metric"] == metric][0][f"r2_{s}"]
        print(f"{metric:<9}{s:<10} n={len(rows):>3}  max|x-mine|={dx:.1e} max|y-mine|={dy:.1e}  R2 from rows {rr:.4f} vs panel {lab:.4f}")
