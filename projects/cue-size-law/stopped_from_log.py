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


def opt(flag, cast, default):
    return cast(toks[toks.index(flag) + 1]) if flag in toks else default


args = {"cue_alpha": size.get("alpha_eff") if size and size.get("alpha_eff") is not None else opt("--cue-alpha", float, 0.0),
        "seed": opt("--seed", int, 0), "steps": opt("--steps", int, 8192), "opt": "adamw", "arm": "cue",
        "cue_dir": "rand", "u_file": opt("--u-file", str, "data/cue_u.npy"), "lr": 1e-4,
        "cue_rows": opt("--cue-rows", str, "0,1,2"), "match_x": opt("--match-x", float, None)}
T = None  # PREREG s3 rule on the frozen-grid curve
for c, n in zip(curve, curve[1:]):
    if c["D"] >= 0.10 and n["D"] >= 0.10:
        T = c["step"]
        break
out = {"args": args, "name": name, "cue_size": size, "dev_curve": curve, "gap0": curve[0]["gap"], "T": T,
       "T_dense": None, "censored": None, "stopped_by_hand": True, "steps_run": curve[-1]["step"],
       "train_s": curve[-1].get("s"), "note": "stopped by hand at the room's decision (k=9: Felix #19448; dir1115: Felix #19557); rebuilt from the log; dense evals and the test read are not in the log",
       "test": {"gap": None}}
(HERE / "data" / "runs" / f"{name}.json").write_text(json.dumps(out))
print(name, "T", T, "steps_run", out["steps_run"], "last D", curve[-1]["D"], "max D", max(c["D"] for c in curve))
