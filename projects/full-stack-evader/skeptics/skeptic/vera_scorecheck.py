"""Vera: score-level rebuild from build's saved per-statement scores (data/scores/<state>.npz), own arithmetic.
Per monitor: line per state (2nd-highest of 200 line-setting true), pooled eval catch over the given states, my city
bootstrap (seed 21, 2000), eval false alarms, eval AUROC per state, and the threshold-resplit null (line recomputed on
2000 bootstrap resamples of the 200 line-setting true statements, seed 1021; 2.5-97.5% of the pooled catch).
    python vera_scorecheck.py <tag> <state> [...]
"""
import json
import sys

import numpy as np

from vera_core import EVAL, EVF, HERE, LINE_T, PROJ, YF, auroc, flags, pooled

TAG, STATES = sys.argv[1], sys.argv[2:]
Z = {s: np.load(PROJ / f"data/scores/{s}.npz") for s in ["clean"] + STATES}
LT = np.where(LINE_T)[0]
rng = np.random.default_rng(1021)
RS = rng.integers(0, 200, (2000, 200))
res = {}
for m in Z["clean"].keys():
    if not all(m in Z[s] and Z[s][m].shape == (2992,) for s in ["clean"] + STATES):
        continue
    r = {"clean": pooled([flags(Z["clean"][m])])["catch"]}
    if STATES:
        r.update(pooled([flags(Z[s][m]) for s in STATES]))
        r["per_seed"] = [pooled([flags(Z[s][m])])["catch"] for s in STATES]
        r["auroc"] = [auroc(Z[s][m][EVAL], YF[EVAL]) for s in STATES]
        # threshold resplit null
        tot = np.zeros(2000)
        for s in STATES:
            sc = Z[s][m]; t2 = np.sort(sc[LT][RS], 1)[:, -2]
            tot += (sc[EVF][None, :] > t2[:, None]).sum(1)
        tot /= len(EVF) * len(STATES)
        r["resplit"] = [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))]
    res[m] = r
json.dump(res, open(HERE / f"vera_scorecheck_{TAG}.json", "w"), indent=1, default=float)
order = [f"tok{i}" for i in range(1, 30)] + [f"mean{i}" for i in range(1, 30)] + ["mlp14", "mega", "base"]
for m in order + [k for k in res if k not in order]:
    if m in res:
        r = res[m]
        print(f"{m:8s} clean {r['clean']:.3f}" + (f" att {r['catch']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] fa {r['fa']:.3f} seeds {np.round(r['per_seed'], 3).tolist()} auroc {np.round(r['auroc'], 3).tolist()} resplit [{r['resplit'][0]:.3f},{r['resplit'][1]:.3f}]" if STATES else ""))
