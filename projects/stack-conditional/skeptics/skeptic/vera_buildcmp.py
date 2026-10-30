"""Compare my own per-prompt P(X) with build's data/eval jsonl (after my numbers were posted)."""
import json, sys, numpy as np
E = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data/eval"
for mine, run, step in [(a.split("=")[0], *a.split("=")[1].split("@")) for a in sys.argv[1:]]:
    z = np.load(mine); te = z["split"] == "test"; ids = list(z["ids"][te])
    for b, hb in (("eager", "eager"), ("comp", "compiled"), ("emu", "emu")):
        rows = {json.loads(l)["id"]: json.loads(l)["px"] for l in open(f"{E}/{run}/{hb}_{step}.jsonl")}
        his = np.array([rows[i] for i in ids]); my = z[f"{b}_px"][te]
        print(f"{run} {step} {hb:8s} mine {my.mean():.4f} build {his.mean():.4f} max|diff| {np.abs(my-his).max():.2e}")
