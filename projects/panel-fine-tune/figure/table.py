"""Print the post's per-probe table (markdown) from data/results/results.json. usage: python figure/table.py"""
import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent.parent / "data/results/results.json").read_text())
A = R["arms"]
lam0 = "panel_lam0" if "panel_lam0" in A else None


def p(v):
    v *= 100
    return f"{v:.1f}%" if 0 < v < 10 else f"{v:.0f}%"


def ci(m):
    lo, hi = m["ci95"]
    return f"{p(m['catch'])} ({p(lo)} to {p(hi)})"


cols = ["Probe", "Original model", "Layer-18 fine-tune", "Panel fine-tune"] + (["Panel fine-tune, no penalty"] if lam0 else [])
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
keys = [f"tok{l}" for l in range(10, 30)] + ["panelmax", "panelmax19", "panelmax_no28"]
names = {"tok29": "layer 28+", "panelmax": "**panel (20 probes)**", "panelmax19": "panel without 28+", "panelmax_no28": "panel without 28"}
for k in keys:
    row = [names.get(k, f"layer {k[3:]}"), p(A["clean"]["monitors"][k]["catch"]), ci(A["layer18"]["monitors"][k]), ci(A["panel"]["monitors"][k])]
    if lam0:
        row.append(p(A[lam0]["monitors"][k]["catch"]))
    print("| " + " | ".join(row) + " |")
