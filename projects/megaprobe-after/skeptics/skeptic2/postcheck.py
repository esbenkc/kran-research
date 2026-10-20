#!/usr/bin/env python3
"""Hugo post-number check for PR #62 (megaprobe-after.md).

Own code. Reuses only my own helpers in score_wide_check.py. Recomputes from saved
score arrays: the full 28-layer gap table, the country-token catch table (D3),
the all-layer sentence-average probe (D2 arm B), fit-seed refits, megaprobe
held-out false alarms, and the held-out-line sensitivity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from score_wide_check import (PROJECT, OUT, STATES, catch_far, dprime, dprime_boot_ratio,  # noqa: E402
                              make_pairs, pooled_city_ci)

D = PROJECT / "data"


def pooled(arrs, true, split, city, n_boot=1000):
    return pooled_city_ci(arrs, true, split, city, n_boot=n_boot)


def main() -> int:
    z = np.load(D / "scores.npz", allow_pickle=True)
    true = 1 - z["y"].astype(int)
    split = z["split"].astype(str)
    city = z["city"].astype(str)
    src = z["src"].astype(str)
    ev = split == "eval"
    out = {}

    # 1. Gap table, all 28 layers (post lines 103-132).
    pairs = make_pairs(true, split, city, src)
    gap = {}
    for l in range(1, 29):
        clean = z[f"clean__lin{l}"]
        cd = dprime(clean, clean, true, pairs)
        ad = float(np.mean([dprime(z[f"{s}__lin{l}"], clean, true, pairs) for s in STATES]))
        row = {"clean": round(cd, 2), "att": round(ad, 2)}
        if l >= 8:
            r = dprime_boot_ratio(l, z, true, pairs, city, n_boot=300)
            row.update(kept=round(r["retention"], 3), lo=round(r["lo"], 3), hi=round(r["hi"], 3))
        gap[l] = row
    out["gap_table"] = gap
    out["upper_all_beat_14_point"] = all(gap[l]["kept"] > gap[14]["kept"] for l in range(15, 29))
    out["upper_any_recovers_lo>=0.5"] = any(gap[l]["lo"] >= 0.5 for l in range(15, 29))

    # 2. Megaprobe per seed, held-out false alarms, fit-seed refits.
    mega = {s: catch_far(z[f"{s}__mega"], true, split, s) for s in ["clean"] + STATES}
    out["mega_seed_catch"] = {s: round(v[0], 4) for s, v in mega.items()}
    out["mega_seed_far"] = {s: round(v[1], 4) for s, v in mega.items()}
    zc = np.load(D / "scores_controls.npz", allow_pickle=True)
    zd = np.load(D / "scores_d3.npz", allow_pickle=True)
    for fit in ["mega_fit1021", "mega_fit1022"]:
        out[f"pooled_{fit}"] = pooled({s: zd[f"{s}__{fit}"] for s in STATES}, true, split, city)

    # 3. Held-out-line sensitivity (post line 162): line at the 1% point of 596 eval true.
    sens = {}
    for name, fn in {
        "quantile_0.99": lambda t: np.quantile(t, 0.99),
        "kth_6": lambda t: np.sort(t)[-6],
        "kth_5": lambda t: np.sort(t)[-5],
        "kth_7": lambda t: np.sort(t)[-7],
    }.items():
        vals = []
        for s in STATES:
            sc = z[f"{s}__mega"]
            line = fn(sc[ev & (true == 1)])
            vals.extend((sc[ev & (true == 0)] > line).tolist())
        sens[name] = round(float(np.mean(vals)), 4)
    out["heldout_line_variants"] = sens

    # 4. D2 arm B: all-layer sentence-average linear probe.
    out["linB_all_clean"] = round(catch_far(zc["clean__linB_all"], true, split, "c")[0], 4)
    out["linB_all_pooled"] = pooled({s: zc[f"{s}__linB_all"] for s in STATES}, true, split, city)
    for arm in ["megaA_l18", "megaA_l22"]:
        out[f"{arm}_clean"] = round(catch_far(zc[f"clean__{arm}"], true, split, "c")[0], 4)
        out[f"{arm}_pooled"] = round(float(np.mean([catch_far(zc[f"{s}__{arm}"], true, split, s)[0] for s in STATES])), 4)

    # 5. D3 country-token probe, every layer (post lines 166-170).
    d3 = {}
    for l in range(1, 29):
        cc, cfar, _, _ = catch_far(zd[f"clean__tok{l}"], true, split, "c")
        per = [catch_far(zd[f"{s}__tok{l}"], true, split, s) for s in STATES]
        d3[l] = {"clean": round(cc, 4), "att": round(float(np.mean([p[0] for p in per])), 4),
                 "far_max": round(max(p[1] for p in per), 4), "far_seeds": [round(p[1], 4) for p in per]}
    for l in [14, 18, 21]:
        d3[l]["ci"] = pooled({s: zd[f"{s}__tok{l}"] for s in STATES}, true, split, city)
    out["d3"] = d3
    out["d3_clean_min_l10_28"] = min(d3[l]["clean"] for l in range(10, 29))

    (OUT / "postcheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
