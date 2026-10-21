"""Vera: behaviour bar (ii) from build's per-position KL file data/acts/<state>/kl.npy: mean over the 1192 eval
statements, every stored position; also max per statement and the share of positions above 0.05.
    python vera_klnpy.py <state> [...]
"""
import json
import sys

import numpy as np

from vera_core import ACTS, EV_IDX, HERE

res = {}
for s in sys.argv[1:]:
    kl = np.load(ACTS / s / "kl.npy"); o = np.load(ACTS / s / "off.npy")
    ev = np.concatenate([kl[o[i]:o[i + 1]] for i in EV_IDX])
    res[s] = {"n_pos": int(len(ev)), "mean": float(ev.mean()), "p99": float(np.percentile(ev, 99)),
              "max": float(ev.max()), "share_gt_0.05": float((ev > 0.05).mean()), "len_matches_tok": bool(len(kl) == o[-1])}
    print(s, res[s], flush=True)
json.dump(res, open(HERE / "vera_klnpy.json", "w"), indent=1)
