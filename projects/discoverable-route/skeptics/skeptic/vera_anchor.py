"""Per-model Delta CIs (runbook - named, seeds 0-39) and runbook / named vs the 1102 open_c4 anchor (unpaired, equal model weight).
Also: linked openers by timing of the storage.md read vs the first denial. usage: python3 vera_anchor.py <tags> <1102 tags>"""
import json, sys
import numpy as np
t = [json.loads(l) for l in open(sys.argv[1])]; old = [json.loads(l) for l in open(sys.argv[2])]
T = {(r["model"], r["cell"], r["seed"]): r for r in t}; M = sorted({r["model"] for r in t})
rng = np.random.default_rng(0); B = 10000
for m in M:
    n_ = np.array([T[(m, "named", s)]["key"] for s in range(40)], float); r_ = np.array([T[(m, "runbook", s)]["key"] for s in range(40)], float)
    I = rng.integers(0, 40, (B, 40)); d = r_[I].mean(1) - n_[I].mean(1)
    print(f"{m:28s} Delta {r_.mean() - n_.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
for cell in ("runbook", "named"):
    dd = np.zeros(B); pt = []
    for m in M:
        a = np.array([T[(m, cell, s)]["key"] for s in range(40)], float)
        o = np.array([x["key"] for x in old if x["model"] == m and x["cell"] == "open_c4"], float)
        dd += a[rng.integers(0, len(a), (B, len(a)))].mean(1) - o[rng.integers(0, len(o), (B, len(o)))].mean(1)
        pt.append(a.mean() - o.mean())
    dd /= len(M)
    print(f"{cell} (this run) - 1102 open_c4 anchor: {np.mean(pt):+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}] (unpaired, different day)")
op = [T[(m, "linked", s)] for m in M for s in range(30) if T[(m, "linked", s)]["opened"]]
print(f"linked openers {len(op)}: read storage.md after first denial {sum(r['rd_after_den'] for r in op)}")
