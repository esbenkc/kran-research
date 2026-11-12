"""Paired read for C3 (Kasper #20657): the below-floor run (nocast x 0.0219, seed 0) and the no-cue run (alpha 0,
seed 0) share the seed and the init, so their trajectories match until the cue matters. Score BOTH runs' snapshots
with the SAME candidate set (the x 0.0219 cue as fed: fp32, no cast; the 7 planted keys; the 1,000 nulls) and
compare the rank of every key at every common snapshot. A cue effect shows as u1110 (the trained key) improving in
the cued run only; a seed effect moves u1110 in both runs, or moves the other keys too.
    NC_DEV=cpu /Users/esben/work/theory-1112-weight-only/env/.venv/bin/python paired_c3.py -> data/paired_c3.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import torch

from audit import KEYS, Spaces, candidates, null_dirs, run_spec, score_adapter, summarize

HERE = Path(__file__).parent
CUED, NULL = "nocast_x0.0219_s0", "c2_a0_s0"


def main():
    sp = Spaces()
    base = sp.base()
    meta = json.load(open(HERE / "raw" / "runs" / CUED / "args.json"))
    spec = run_spec(meta)
    names, C = candidates(sp, spec, base, null_dirs())
    steps = sorted(set(int(p.stem[4:]) for p in (HERE / "raw" / "runs" / CUED).glob("step*.pt"))
                   & set(int(p.stem[4:]) for p in (HERE / "raw" / "runs" / NULL).glob("step*.pt")))
    out = {"spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}, "steps": {}}
    for st in steps:
        row = {}
        for run in (CUED, NULL):
            ad = torch.load(HERE / "raw" / "runs" / run / f"step{st}.pt", map_location="cpu")
            s, _ = summarize(score_adapter(ad, spec, C), names)
            q = s["qkv26"]
            row[run] = {stat: {"true_rank": q[stat]["rank"], "keys": {k: v["rank"] for k, v in q[stat]["keys"].items()}}
                        for stat in ("top1", "gain", "aonly")}
        out["steps"][str(st)] = row
        print(st, "gain u1110 cued", row[CUED]["gain"]["true_rank"], "null", row[NULL]["gain"]["true_rank"], flush=True)
    (HERE / "data" / "paired_c3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
