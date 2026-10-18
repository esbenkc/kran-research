"""Lambda pilot on seed 99 (not scored). Rule (plan #2673/#2687, posted before the pilot):
walk the pilot grid largest-first; a value 'hits the stop' if at some check the STOP-set TPR at the clean
threshold t0 at the FINAL step is >= 50 points below its clean value (PREREG). l3 = the largest value that hits the stop and whose
next-larger value does not (bracketed): if the top grid value already hits, step up by sqrt(10) until a miss.
Frozen grid = {0, l3/10, l3/sqrt(10), l3}. The pilot runs the full step budget (same as the scored runs).
"""
import json
import subprocess
import sys

import numpy as np

from common import DATA, threshold

GRID = [10000, 3162, 1000, 316, 100, 32, 10, 3.2, 1]
OUT = DATA / "pilot"


def stop_drop(stem):
    z = np.load(OUT / f"{stem}.npz")
    y, sp, S = z["y"], z["split"], z["scores"].astype(np.float64)
    t0 = threshold(S[0][(sp == "refit") & (y == 0)])
    m = (sp == "stop") & (y == 1)
    tpr = (S[:, m] > t0).mean(1)
    return [int(c) for c in z["checks"]], (tpr[0] - tpr).tolist()


def trial(lam, log):
    stem = f"lam{lam:g}__seed99"
    if not (OUT / f"{stem}.npz").exists():
        subprocess.run([sys.executable, "attack.py", "--lam", str(lam), "--seed", "99", "--tag", "pilot"], check=True)
    checks, drop = stop_drop(stem)
    rec = json.loads((OUT / f"{stem}.json").read_text())
    hit = drop[-1] >= 0.5  # PREREG: FINAL drop (step N), on the stop-set positives (never eval)
    log.append({"lam": lam, "hit": hit, "max_drop_stop": max(drop), "final_drop_stop": drop[-1],
                "first_hit_step": next((c for c, d in zip(checks, drop) if d >= 0.5), None),
                "retain_kl_heldout": rec["retain_kl_heldout"], "seconds": rec["seconds"]})
    print(json.dumps(log[-1]), flush=True)
    return hit


CAP = 1.01e6  # Felix #2733 (15:36 CEST): at most 2 pilot values after 1e5; if ~1e6 still hits, l3 = 1e6, no miss found


def sig3(v):
    return float(f"{v:.3g}")


def main():
    log = []
    l3, capped = None, False
    if trial(GRID[0], log):
        lam = GRID[0]
        while True:
            up = round(lam * 10 ** 0.5)
            if up > CAP:
                l3, capped = lam, True
                break
            if not trial(up, log):
                l3 = lam
                break
            lam = up
    else:
        for lam in GRID[1:]:
            if trial(lam, log):
                l3 = lam
                break
    # scored grid values rounded to 3 significant digits (pilot values carry sqrt(10) rounding, e.g. 1000004)
    grid = None if l3 is None else [0, sig3(l3 / 10), sig3(l3 / 10 ** 0.5), sig3(l3)]
    (OUT / "pilot.json").write_text(json.dumps({"rule": __doc__, "cap_rule": "Felix #2733: stop after the value ~1e6; "
                                                "if it hits, l3 = 1e6 and no miss was found up to 1e6",
                                                "pilot_grid": GRID, "log": log, "l3": l3, "capped_no_miss": capped,
                                                "frozen_grid": grid}, indent=1))
    print("frozen grid:", grid)


if __name__ == "__main__":
    main()
