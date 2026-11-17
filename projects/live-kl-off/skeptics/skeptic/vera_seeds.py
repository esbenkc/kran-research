"""Vera: seed-level summary across KL-off seeds, from raw reads only (uses vera_check.py).

usage: vera_seeds.py TEST_POOL [--placebo PLACEBO_POOL PLACEBO_READS_SUBDIR] RUN_DIR [RUN_DIR ...]
Prints per-seed curve, onset, s_P at the common read (smallest last read over seeds), mean and
t-interval over seeds, and s_R at each seed's last read.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from vera_check import run

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

args = sys.argv[1:]
pool = args.pop(0)
runs = args
res = {}
for r in runs:
    out, onset = run(r, pool)
    res[r] = dict(reads={o["step"]: o for o in out}, onset=onset)
    last = max(res[r]["reads"])
    lo = res[r]["reads"][last]
    print(f"{os.path.basename(r)}: reads {sorted(res[r]['reads'])} onset {onset} last {last} "
          f"s_P {lo['s_P']:+.4f} [{lo['s_P_ci'][0]:+.4f},{lo['s_P_ci'][1]:+.4f}] s_R {lo['s_R']:+.4f} "
          f"[{lo['s_R_ci'][0]:+.4f},{lo['s_R_ci'][1]:+.4f}] cheat_off {lo['cheat_off']:.3f}")
common = min(max(v["reads"]) for v in res.values())
vals = [v["reads"][common]["s_P"] for v in res.values()]
m = float(np.mean(vals))
if len(vals) > 1:
    se = float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
    h = T975[len(vals) - 1] * se
    print(f"seed-level s_P at common read {common}: {m:+.4f} [{m-h:+.4f}, {m+h:+.4f}] (t over {len(vals)} seeds) vals {vals}")
else:
    print(f"seed-level s_P at common read {common}: {m:+.4f} (1 seed, no interval)")
json.dump({os.path.basename(k): dict(onset=v["onset"], reads=list(v["reads"].values())) for k, v in res.items()},
          open(os.path.join(os.path.dirname(__file__), "vera_seeds_out.json"), "w"), indent=1, default=float)
