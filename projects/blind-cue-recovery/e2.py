"""E2 (PREREG.md section E2, frozen 06:32 PST before compute; exploratory, no bar, never changes the status):
does the weight read lead behaviour when the switch is slow? No new training: the parent's seed-0 sweep adapters.

Runs (x, T): sw_a0.00390625 (4.20, 192), sw_a0.00195312 (2.06, 320), sw_a0.000976562 (1.05, 704),
sw_a0.000488281 (0.425, 1728), sw_a0.000244141 (0.217, 4352). Reads T/4, T/2 (the E2 read), T.
Statistics, no reference: RAW top-1 e1, and E1-RAW (project off the target's own top input direction).
BEATS CHANCE = (a) max of the 1,000 random cues at the run's alpha and cast, (b) max of the true cue on the clean adapters
c2_a0_s0 and c2_a0_s1 at the saved snapshot nearest to t in log updates (c2_a0_s1 only up to 2,048), treated the same way,
(c) the slice-only null p95 from the run's A_t (projected the same way for E1-RAW).
Lead claim allowed only if RAW, or E1-RAW, on its own beats chance at T/2 in >= 3 of 5 sizes (amendment 6da9ddccb).
    NC_DEV=cpu python e2.py -> data/e2.json; stdout = the frozen count only
"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from blind import Scorer  # noqa: E402

RUNS = [("sw_a0.00390625_s0", 192), ("sw_a0.00195312_s0", 320), ("sw_a0.000976562_s0", 704),
        ("sw_a0.000488281_s0", 1728), ("sw_a0.000244141_s0", 4352)]
CLEAN = {"c2_a0_s0": 20480, "c2_a0_s1": 2048}


def snaps(run):
    return sorted(int(p.stem[4:]) for p in (HERE / "raw" / "runs" / run).glob("step*.pt") if p.stem[4:] != "0")


def nearest(run, t):
    s = [x for x in snaps(run) if x <= CLEAN[run]]
    return min(s, key=lambda x: (abs(math.log(x) - math.log(t)), x))


sc = Scorer()
out = []
for run, T in RUNS:
    meta = json.load(open(HERE / "data" / "runs" / f"{run}.json"))
    ev = {c["step"]: c["D"] for c in meta.get("dev_curve", []) + meta.get("dense_curve", [])}
    assert meta["T"] == T, (run, meta["T"])
    for label, t in [("T/4", T // 4), ("T/2", T // 2), ("T", T)]:
        tp = f"raw/runs/{run}/step{t}.pt"
        assert (HERE / tp).exists(), tp
        w = sc.top_dir([tp])
        rec = {"run": run, "x": meta["cue_size"]["x"], "T": T, "read": label, "t": t, "D": ev.get(t)}
        for stat in ["RAW", "E1-RAW"]:
            if stat == "RAW":
                row, _ = sc.score(tp, [], run)
                sl = sc.slice_null(tp, run)
            else:
                row, _ = sc.score_proj(tp, w, run)
                sl = sc.slice_null_proj(tp, w, run)
            held = []
            for c in CLEAN:
                cp = f"raw/runs/{c}/step{nearest(c, t)}.pt"
                if stat == "RAW":
                    h, _ = sc.score(cp, [], run)
                else:
                    h, _ = sc.score_proj(cp, sc.top_dir([cp]), run)
                held.append({"clean": cp, "e1": h["e1_true"]})
            p95 = float(np.quantile(sl, 0.95))
            hmax = max(x["e1"] for x in held)
            a, b, c_ = row["e1_true"] > row["null_max"], row["e1_true"] > hmax, row["e1_true"] > p95
            rec[stat] = {"e1_true": row["e1_true"], "rank": row["rank"], "null_max": row["null_max"], "held": held,
                         "held_max": hmax, "slice_p95": p95, "beats_a": a, "beats_b": b, "beats_c": c_,
                         "beats_chance": bool(a and b and c_)}
        out.append(rec)
(HERE / "data" / "e2.json").write_text(json.dumps(out, indent=1))
half = [r for r in out if r["read"] == "T/2"]
# amended rule (PREREG E2 amendment, commit 6da9ddccb, 06:32:36, before compute): each read counted on its own
for stat in ["RAW", "E1-RAW"]:
    n = sum(1 for r in half if r[stat]["beats_chance"])
    print(f"E2: {stat} beats chance at T/2 in {n} of 5 sizes (lead wording for this read allowed if >= 3)")
