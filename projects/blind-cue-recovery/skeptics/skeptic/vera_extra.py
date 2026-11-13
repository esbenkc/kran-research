"""Vera: SM replicate cells (single twins, split halves) at t_h, and the 1-twin curve incl. a positive control at 256.

Positive control: RAW ranks the cue 1st at 256 for m1 (u1111) and m3 (u1113). If subtracting a twin keeps it 1st there,
the SM pipeline does not destroy a cue that is present; the t_h miss is then about timing, not a bug.
"""
import glob
import json
import os

import numpy as np

from vera_check import CUES, NA, RUNS, TH, e1, load, remainder, target_path, v1

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
for i in range(6):
    t = TH[i]
    paths = sorted(glob.glob(f"{RUNS}/sm_m{i}_t{t}_d*/step{t}.pt"))
    tw = [load(p) for p in paths]
    sd = load(target_path(i, t))
    cells = {f"single_{os.path.basename(os.path.dirname(p))[-4:]}": [x] for p, x in zip(paths, tw)}
    cells["half_12"], cells["half_34"] = tw[:2], tw[2:]
    for k, refs in cells.items():
        v = v1(remainder(sd, refs))[0]
        et = float(e1(v, CUES[i])[0])
        rank = int(1 + (e1(v, NA) >= et).sum())
        res[f"m{i}|t_h|{t}|{k}"] = {"e1": et, "rank_a": rank}
    print(f"m{i} t_h={t}: " + " ".join(f"{k}:{res[f'm{i}|t_h|{t}|{k}']['rank_a']}" for k in cells), flush=True)

for i in range(6):
    row = []
    for t in (8, 16, 24, 32, 48, 64, 128, 256):
        p = sorted(glob.glob(f"{RUNS}/sm_m{i}_t{t}_d*/step{t}.pt"))
        if not p:
            continue
        sd = load(target_path(i, t))
        v = v1(remainder(sd, [load(p[0])]))[0]
        et = float(e1(v, CUES[i])[0])
        rank = int(1 + (e1(v, NA) >= et).sum())
        res[f"m{i}|curve|{t}|1twin"] = {"e1": et, "rank_a": rank, "twin": os.path.basename(os.path.dirname(p[0]))}
        row.append(f"{t}:{rank}")
    if row:
        print(f"m{i} 1-twin curve: " + " ".join(row), flush=True)
json.dump(res, open(f"{OUT}/vera_extra.json", "w"), indent=1)
