"""Gap over training (Tove #13851 panel): cached-input dGap on the 200 test prompts at every saved checkpoint.
dGap = mean P(X | serving cache) - mean P(X | grading cache) through the checkpoint's merged tail, minus the untrained
model's cached gap (data/base_cached_test.json). Sources: data/runs/<run>[_at<step>].json (rl_adapt.py finish()),
twins from the parent stack-conditional runs (same cached-tail reading, its data/runs/stack_s<k>_lr0.0001_kl1*.json).
    PY gap_curves.py   -> data/gap_curves.json
"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
PARENT = Path("/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data/runs")
b = json.load(open(HERE / "data/base_cached_test.json"))["gap"]
def dg(p):
    t = json.load(open(p))["test_px_cached"]
    return float(np.mean(t["compiled"]) - np.mean(t["eager"])) - b
out = {"note": "cached-input reading, 200 test prompts, minus untrained cached gap " + f"{b:+.4f}", "runs": {}}
runs = {"adv_mid_s0": ("hide", HERE / "data/runs", "adv_mid_s0", 8192), "adv_mid_s1": ("hide", HERE / "data/runs", "adv_mid_s1", 8192),
        "adv_mid_s0x": ("hide", HERE / "data/runs", "adv_mid_s0x", 16384), "adv_mid_s1x": ("hide", HERE / "data/runs", "adv_mid_s1x", 16384),
        "cln_s0": ("clean", HERE / "data/runs", "cln_s0", 8192), "cln_s1": ("clean", HERE / "data/runs", "cln_s1", 8192),
        "cln_s2": ("clean", HERE / "data/runs", "cln_s2", 8192), "pilot_cap1_s0": ("cap-only", HERE / "data/runs", "pilot_cap1_s0", 4096),
        "stack_s0": ("plain", PARENT, "stack_s0_lr0.0001_kl1", 8192), "stack_s1": ("plain", PARENT, "stack_s1_lr0.0001_kl1", 8192)}
for k, (kind, d, name, final) in runs.items():
    pts = []
    for s in range(1024, final + 1, 1024):
        p = d / (f"{name}.json" if s == final else f"{name}_at{s}.json")
        if p.exists():
            pts.append([s, round(dg(p), 5)])
    if pts:
        out["runs"][k] = {"kind": kind, "points": pts, "source": str(d.relative_to(d.parents[2])) if d != PARENT else "parent stack-conditional data/runs"}
        print(k, kind, pts[-3:])
json.dump(out, open(HERE / "data/gap_curves.json", "w"), indent=1)
