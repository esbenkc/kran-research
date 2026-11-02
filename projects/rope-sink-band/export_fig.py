"""Main-figure data (Owain): data/fig_main.json.
left:  per-pair share of the first token's lead, G_j / sum G (calibration only), pair index 0..63, band flags,
       location edges 21 (base 1e6, 512 tokens) and 46 (base 1e4, 4096 tokens).
right: one row per removed set: kind (band / random / energy / slowest8), calibration share of the lead
       (sum_{j in set} G_j / sum G), whole-model cut per held-out set and pooled, direct per-layer cut (set A, B),
       loss change. Pooled = all 256 prompts of A and B."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
tag = "Qwen3-1.7B"
OUT = HERE / "data" / tag
band = json.load(open(OUT / "band.json")); res = json.load(open(OUT / "results.json"))
G = np.array(band["G"]); share = G / G.sum()
sets = {"P": ("band", band["B"])}
sets.update({f"R{i+1:02d}": ("random", r) for i, r in enumerate(band["R"])})
sets.update({f"E{i+1:02d}": ("energy", e) for i, e in enumerate(band["E"])})
sets["LOW8"] = ("slowest8", band["LOW_8"])
rows = []
for name, (kind, pairs) in sets.items():
    row = {"name": name, "kind": kind, "pairs": pairs, "share": float(share[pairs].sum())}
    for s, d in res["sets"].items():
        a = d["arms"].get(name)
        if a:
            row[f"cut_{s}"] = round(a["cut"], 4); row[f"dloss_{s}"] = round(a["dloss"], 4)
        dd = d.get("direct")
        if dd and name == "P":
            row[f"direct_{s}"] = round(dd["cut_P_all_layers"], 4)
        elif dd and kind == "random":
            row[f"direct_{s}"] = round(dd["cut_R_all_layers"][int(name[1:]) - 1], 4)
    rows.append(row)
fig = {"model": tag, "B": band["B"], "k": band["k"], "status": res.get("status"),
       "bars": {"yes": 0.70, "no": 0.50}, "edges": {"a_base1e6_L512": 21, "b_base1e4_L4096": 46, "c_base1e6_L32768": 40},
       "left": [{"pair": j, "share": round(float(share[j]), 5), "in_band": j in band["B"]} for j in range(64)],
       "right": rows,
       "baseline": {s: {"S": round(d["S_base"], 4), "loss": round(d["loss_base"], 4), "n": d["n"]} for s, d in res["sets"].items()}}
json.dump(fig, open(HERE / "data" / "fig_main.json", "w"), indent=1)
print("rows", len(rows), "with cut_A:", sum("cut_A" in r for r in rows), "cut_B:", sum("cut_B" in r for r in rows))
