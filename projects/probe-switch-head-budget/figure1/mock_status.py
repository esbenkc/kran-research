"""Mock data/status.json for a mock tree, from the mock's own numbers (tests gen.py's cross-check path). NOT data."""
import json, glob, statistics, sys
P = sys.argv[1]
cells = {}
for f in sorted(glob.glob(P + "/data/kaudit/k6_s*.json")):
    a = json.load(open(f)); n = a["run"]; tc = a["test_cut"]
    S0 = a["test_base"]["strict32"]
    cells[n] = {"live": S0 >= 0.75, "fp32": {"S0": S0, "S8": tc["greedy_TB_k8"]["strict32"],
                "R8": statistics.median(tc[f"rand8_{j:02d}"]["strict32"] for j in range(50))}}
counted = [n for n, c in cells.items() if c["live"]]
ok3 = len(counted) == 3
st = ("CONFIRMED" if ok3 and all(cells[n]["fp32"]["S8"] < .25 and cells[n]["fp32"]["R8"] >= .75 for n in counted)
      else "REJECTED" if ok3 and any(cells[n]["fp32"]["S8"] >= .25 for n in counted) else "UNCONFIRMED")
json.dump({"status": st, "counted_k6": counted, "notes_k6": [], "cells": cells}, open(P + "/data/status.json", "w"))
print(P, st, counted)
