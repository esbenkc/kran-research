"""Vera: phase 0 (RAW arm) from the parent's saved snapshots, own code (vera_lib)."""
import json
import os

import numpy as np

from vera_lib import CUES, cue_diff, e1, load, nulls, remainder, v1

PARENT = os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit/raw/runs")
OUT = os.path.dirname(os.path.abspath(__file__))
NA = nulls(1000, 20261112)  # frozen null seed (as audit.null_dirs)
NB = nulls(1000, 7)  # null-of-null: a different seed
res = {}
for i in range(6):
    run = f"dir_u111{i}_a0.03125_s{i}"
    for t in (8, 16, 24, 32, 64, 256):
        v, s = v1(remainder(load(f"{PARENT}/{run}/step{t}.pt"), []))
        et = float(e1(v, CUES[i])[0])
        ea, eb = e1(v, NA), e1(v, NB)
        res[f"{run}@{t}"] = {"e1": et, "rank": int(1 + (ea >= et).sum()), "rank_null7": int(1 + (eb >= et).sum()),
                             "max_null": float(ea.max()), "s1_s2": [float(s[0]), float(s[1])]}
        r = res[f"{run}@{t}"]
        print(f"{run}@{t:>3} e1 {et:.4f} max null {r['max_null']:.4f} rank {r['rank']:>4} (null7 {r['rank_null7']:>4}) "
              f"s1/s2 {s[0]:.3f}/{s[1]:.3f}")
json.dump(res, open(f"{OUT}/vera_raw.json", "w"), indent=1)
