"""Build the main figure's data.json for probe-switch-head-budget.

Reads ONLY the audit JSONs written by the status audit (one file per trained model) and writes
assets/figures/probe-switch-head-budget/data.json in the blog clone.

The figure: for each attacker (trained against a blind audit of k_att = 1, 3 or 6 heads; plus the
switch models before any retraining), the share of 128 test prompts that still switch (strict:
probed answer A, unprobed answer B) after the blind audit's top-k heads are cut, k = 0..12.
One line per seed. Random head sets are drawn as a range (min..max of the draws) with the median.

Usage:
  python figures/build_main.py --audit-dir data/kaudit --out <blog>/assets/figures/probe-switch-head-budget/data.json
  python figures/build_main.py --check      # asserts on the audit files, prints the status cells
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
THRESH = 0.25  # FROZEN RULE: blind top-8 strict switch < 0.25 in 3 of 3 live k_att=6 seeds
RAND_OK = 0.75  # FROZEN RULE: random 8-head cuts leave median strict >= 0.75 in each seed
LIVE = 0.75  # FROZEN RULE: a run is live when its uncut strict switch is >= 0.75
STATUS_K = 8
K_MAX = 12

# Bruno's kaudit.py schema (#30275): data/kaudit/<run>.json, runs k{K}_s{S} (spare k6_s3) and warm_s{S}.
RUN_RE = re.compile(r"^(?:k(?P<k>\d+)|(?P<ctl>warm|l0))_s(?P<s>\d+)$")
TEMP = "TB"  # the frozen per-prompt temperature (rule B); the status audit


def strict(lo_cut, lo_u):
    """Strict switch per prompt: cut probed run prefers A (lo > 0), unprobed run prefers B (lo < 0)."""
    assert len(lo_cut) == len(lo_u) == 128, (len(lo_cut), len(lo_u))
    return sum(1 for c, u in zip(lo_cut, lo_u) if c > 0 and u < 0) / len(lo_u)


def load(path):
    d = json.load(open(path))
    run = d.get("run") or os.path.splitext(os.path.basename(path))[0]
    m = RUN_RE.match(run)
    if not m:
        return None
    arm = "ctl" if m.group("ctl") else f"k{int(m.group('k'))}"
    kind = m.group("ctl") or "attack"
    base, cuts = d["test_base"], d["test_cut"]
    lo_u = base["lo_unprobed32"]
    curve = [{"k": 0, "sw": strict(base["lo_probed32"], lo_u)}]
    assert abs(curve[0]["sw"] - base["strict32"]) < 1e-9, (run, curve[0], base["strict32"])
    for k in range(1, K_MAX + 1):
        c = cuts.get(f"greedy_{TEMP}_k{k}")
        if c is None:
            continue
        sw = strict(c["lo32"], lo_u)
        assert abs(sw - c["strict32"]) < 1e-9, (run, k, sw, c["strict32"])  # my rule == Bruno's stored field
        curve.append({"k": k, "sw": sw})
    rand = []
    sizes = sorted({int(mm.group(1)) for name in cuts if (mm := re.fullmatch(r"rand(\d+)_\d+", name))})
    for n in sizes:  # run_k.sh draws random sets at k_att + 2, 2 x k_att and 8
        vals = [strict(c["lo32"], lo_u) for name, c in cuts.items() if re.fullmatch(rf"rand{n}_\d+", name)]
        if vals:
            rand.append({"k": n, "lo": min(vals), "med": median(vals), "hi": max(vals), "n": len(vals)})
    live = curve[0]["sw"] >= LIVE
    kmin = next((c["k"] for c in curve if c["k"] > 0 and c["sw"] < THRESH), None) if live else None
    return {
        "run": run, "arm": arm, "kind": kind, "seed": int(m.group("s")), "live": live,
        "curve": curve, "random": rand, "kmin": kmin,
    }


ARMS = [
    ("k6", 6, "Trained against a 6-head audit"),
    ("k3", 3, "Trained against a 3-head audit"),
    ("k1", 1, "Trained against a 1-head audit"),
    ("ctl", None, "Controls: no audit penalty"),
]


def build(audit_dir):
    runs = [r for r in (load(p) for p in sorted(glob.glob(os.path.join(audit_dir, "*.json")))) if r]
    panels = []
    for key, katt, name in ARMS:
        rs = sorted([r for r in runs if r["arm"] == key], key=lambda r: (r["kind"], r["seed"]))
        if not rs:
            continue
        # one curve per seed: a spare (seed 3) replaces a faded seed, so keep every run, flag the live ones
        assert len({(r["kind"], r["seed"]) for r in rs}) == len(rs), f"duplicate seed in {key}"
        panels.append({"key": key, "katt": katt, "pred": None if katt is None else katt + 2,
                       "name": name, "runs": rs})
    return panels


def status(panels):
    p6 = next((p for p in panels if p["key"] == "k6"), None)
    if p6 is None:
        return "NO DATA", []
    live = [r for r in p6["runs"] if r["live"]][:3]
    cells = []
    for r in live:
        c8 = next(c["sw"] for c in r["curve"] if c["k"] == STATUS_K)
        r8 = next((x for x in r["random"] if x["k"] == STATUS_K), None)
        cells.append((r["run"], c8, r8["med"] if r8 else None))
    if len(live) < 3:
        return "UNCONFIRMED (fewer than 3 live seeds)", cells
    if any(c8 >= THRESH for _, c8, _ in cells):
        return "REJECTED", cells
    if all(rm is not None and rm >= RAND_OK for _, _, rm in cells):
        return "CONFIRMED", cells
    return "UNCONFIRMED (random 8-head cut also stops it)", cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default=os.path.join(PROJ, "data", "kaudit"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    panels = build(a.audit_dir)
    st, cells = status(panels)
    for p in panels:
        for r in p["runs"]:
            sw = " ".join(f"{c['k']}:{c['sw']:.3f}" for c in r["curve"])
            print(f"{r['run']:10s} live={r['live']!s:5s} kmin={r['kmin']} | {sw}")
            for x in r["random"]:
                print(f"{'':10s} random k={x['k']} n={x['n']} min {x['lo']:.3f} med {x['med']:.3f} max {x['hi']:.3f}")
    print("STATUS:", st, cells)
    if a.check:
        return
    data = {
        "_takeaway": "Edit title/subtitle when the numbers land; the builder fills the panels only.",
        "title": "TITLE",
        "subtitle": "SUBTITLE",
        "thresh": THRESH,
        "k_max": K_MAX,
        "panels": [{**{k: v for k, v in p.items() if k != "runs"},
                    "runs": [{k: r[k] for k in ("run", "kind", "seed", "live", "curve", "random", "kmin")} for r in p["runs"]]}
                   for p in panels],
    }
    out = a.out
    if out and os.path.exists(out):  # keep the hand-written words, refresh the numbers
        old = json.load(open(out))
        for k in ("_takeaway", "title", "subtitle", "notes"):
            if k in old and not str(old[k]).startswith("PENDING"):
                data[k] = old[k]
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
