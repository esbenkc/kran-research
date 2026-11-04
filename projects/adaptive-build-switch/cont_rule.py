"""Frozen 2x rule (PREREG v1, D16): continue a penalised switch to 16,384 updates when dGap(8,192) < 0.10 AND
dGap(8,192) > dGap(6,144) + 0.02. dGap read from the run's cached test P(X) (200 test prompts, finish() of rl_adapt.py)
minus the untrained cached gap (data/base_cached_test.json). Exit 0 = continue, 1 = stop. Writes data/cont_<run>.json.
    PY cont_rule.py adv_mid_s0
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
run = sys.argv[1]
b = json.load(open(HERE / "data/base_cached_test.json"))["gap"]
def dg(name):
    t = json.load(open(HERE / "data/runs" / f"{name}.json"))["test_px_cached"]
    return float(np.mean(t["compiled"]) - np.mean(t["eager"])) - b
d8, d6 = dg(run), dg(f"{run}_at6144")
cont = d8 < 0.10 and d8 > d6 + 0.02
json.dump({"run": run, "dgap_8192": d8, "dgap_6144": d6, "continue": cont, "rule": "d8 < 0.10 and d8 > d6 + 0.02"},
          open(HERE / f"data/cont_{run}.json", "w"), indent=1)
print(run, "dGap 6144", round(d6, 4), "8192", round(d8, 4), "CONTINUE" if cont else "STOP")
sys.exit(0 if cont else 1)
