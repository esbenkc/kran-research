"""Vera: does the penalised switch keep its build gap UNDER a system turn (75% of its training episodes) while the
plain-prompt dGap fades? Own bf16 tail on build's cached block-26 inputs (raw/cache/sys, eager vs compiled, 12
TRAINING system texts), on the 64 DEV prompts (not RL-trained). Gap per text = mean P(serving) - mean P(grading);
report mean over texts and max, minus the untrained model's same number.
usage: vera_sysgap.py OUT.json label=path[,...]
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
from vera_gap import load, EMB, xids

C = os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/raw/cache/sys")
meta = json.load(open(f"{C}/meta.json"))
dev = np.array([i for i, p in enumerate(meta["prompt_ids"]) if p.startswith("dev")])
assert len(dev) == 64, len(dev)
X = {(b, t): vt.bits_to_t(np.load(f"{C}/{b}_train_t{t}_tail.npy", mmap_mode="r")[dev]) for b in ("eager", "compiled") for t in range(12)}
res = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
for spec in sys.argv[2].split(","):
    lab, path = spec.split("=", 1)
    t0 = time.time(); tl = vt.Tail(load(path), EMB, torch.bfloat16)
    g, s = [], []
    for t in range(12):
        g.append(float(np.mean(tl.px(X["eager", t], xids)))); s.append(float(np.mean(tl.px(X["compiled", t], xids))))
    gaps = [b - a for a, b in zip(g, s)]
    res[lab] = {"grading": g, "serving": s, "gap_mean": float(np.mean(gaps)), "gap_max": float(np.max(gaps))}
    print(lab, f"grading {np.mean(g):.3f} serving {np.mean(s):.3f} sys-gap mean {np.mean(gaps):+.4f} max {np.max(gaps):+.4f}",
          f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
