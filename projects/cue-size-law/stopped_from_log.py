"""Rebuild data/runs/<run>.json for a run STOPPED by hand (Felix #19448: k=9 stopped at the room's decision) from its
log: the frozen-grid evals are printed there ('step N {...}'); dense evals and the test read are not, so they are absent.

    /usr/bin/python3 stopped_from_log.py ext_a6.10352e-05_s0 "--cue-alpha 6.103515625e-05 --seed 0 --steps 45056"
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
name, argline = sys.argv[1], sys.argv[2]
curve, size = [], None
for l in open(HERE / "logs" / f"{name}.log"):
    if l.startswith("step "):
        body = l[l.index("{"):]
        if " size " in body:
            body, sz = body.split(" size ", 1)
            size = ast.literal_eval(sz.strip())
        curve.append(ast.literal_eval(body.strip()))
toks = argline.split()
args = {"cue_alpha": float(toks[toks.index("--cue-alpha") + 1]), "seed": int(toks[toks.index("--seed") + 1]),
        "steps": int(toks[toks.index("--steps") + 1]), "opt": "adamw", "arm": "cue", "cue_dir": "rand",
        "u_file": "data/cue_u.npy", "lr": 1e-4}
out = {"args": args, "name": name, "cue_size": size, "dev_curve": curve, "gap0": curve[0]["gap"], "T": None,
       "censored": None, "stopped_by_hand": True, "steps_run": curve[-1]["step"],
       "train_s": curve[-1].get("s"), "note": "stopped at the room's decision (Felix #19448); rebuilt from the log",
       "test": {"gap": None}}
(HERE / "data" / "runs" / f"{name}.json").write_text(json.dumps(out))
print(name, "steps_run", out["steps_run"], "last D", curve[-1]["D"], "max D", max(c["D"] for c in curve))
