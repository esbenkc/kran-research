"""Figure 1 example picker for the grid (Owain). Rule B panel + logistic reader, same code as gen2.py.
Qualifies: a cities false statement that
  clean model:            panel catches AND reader catches
  every in-loop seed:     panel MISSES AND reader catches
  every 1022 plain seed:  reader catches (panel: reported, preferred missed too)
  and its true twin is flagged by no rule in any state.
Run from projects/recalibrated-panel with the .venv python:  pick2.py [seeds glob for in-loop]
"""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "0"] + sys.argv[1:]  # gen2 reads argv[1] as an example index; neutralise
from pick import EVPOS, R, SRC, ruleB, twin  # noqa: E402

src = (HERE / "gen2.py").read_text().split("rows = []")[0]  # only the data helpers
g2 = {"__file__": str(HERE / "gen2.py")}
exec(compile(src, "gen2_head", "exec"), g2)
reader = g2["reader"]

import os  # noqa: E402

FORM = os.environ.get("FORM", "neg_cities")
INLOOP = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else "data/scores/recal__lam10000__seed*.npz"))
INLOOP = [p for p in INLOOP if "step" not in p]
PLAIN = [f"data/prior1022/scores/panel__lam10000__seed{s}.npz" for s in range(3)]
print("in-loop states:", INLOOP)
st = {p: (ruleB(p), reader(p)) for p in ["data/scores/clean.npz"] + PLAIN + INLOOP}
hit = lambda d, i: d["pm"][i] > d["line"]
out = []
for i in EVPOS:
    if SRC[i] != FORM:
        continue
    t = twin(i)
    b, r = st["data/scores/clean.npz"]
    ok = hit(b, i) and hit(r, i)
    ok &= all((not hit(st[p][0], i)) and hit(st[p][1], i) for p in INLOOP)
    ok &= all(hit(st[p][1], i) for p in PLAIN)
    ok &= not any(hit(d, t) for p in st for d in st[p])
    if ok:
        plain_missed = all(not hit(st[p][0], i) for p in PLAIN)
        out.append((plain_missed, i))
print(len(out), "qualify;", sum(m for m, _ in out), "also missed by the panel in all 3 plain runs")
for m, i in sorted(out, key=lambda x: -x[0]):
    print(("BOTH " if m else "     ") + str(i), R[i]["text"], "| twin:", R[twin(i)]["text"])
