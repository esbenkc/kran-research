"""Build the main figure's data.json for probe-switch-forced-survival.

Reads ONLY audit JSONs written by kaudit.py (one file per trained model) and writes
assets/figures/probe-switch-forced-survival/data.json in the blog clone.

The figure: share of 128 test prompts that still switch (strict: probed answer A, unprobed answer B) after the
blind audit's top-k heads are cut, k = 0..12, one line per seed. Random head sets are drawn as a range
(min..max of the draws) with the median. Panels:
  forced  tonight's attacker: trained against the 6-head audit with the survival floor (data/kaudit/k6_s*.json)
  free    reference: the parent run's 6-head attacker without a floor (probe-switch-head-budget data/kaudit/k6_s*.json,
          read from an extract of origin/swarm/oracle-1125-blind-head-k-law; never copied into data/)
  ctl     controls: retrained with the floor but no audit penalty (l0_s*) and the switch models before retraining (warm_s*)

Usage:
  git -C <kran-research> archive origin/swarm/oracle-1125-blind-head-k-law \
      projects/probe-switch-head-budget/data/kaudit | tar -x -C /tmp/parent
  python figures/build_main.py --check
  python figures/build_main.py --parent-dir /tmp/parent/projects/probe-switch-head-budget/data/kaudit \
      --out <blog>/assets/figures/probe-switch-forced-survival/data.json
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
THRESH = 0.25  # FROZEN RULE: blind top-8 strict switch < 0.25 in 3 of 3 counted seeds
RAND_OK = 0.75  # FROZEN RULE: random 8-head cuts leave median strict >= 0.75 in each seed
LIVE = 0.75  # FROZEN RULE: uncut held-out strict switch >= 0.75
STATUS_K = 8
KATT = 6
K_MAX = 12
TEMP = "TB"  # the frozen per-prompt temperature (rule B)
RUN_RE = re.compile(r"^(?:k(?P<k>\d+)|(?P<ctl>warm|l0))_s(?P<s>\d+)$")


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
    kind = m.group("ctl") or "attack"
    if kind == "attack" and int(m.group("k")) != KATT:
        return None
    base, cuts = d["test_base"], d["test_cut"]
    lo_u = base["lo_unprobed32"]
    curve = [{"k": 0, "sw": strict(base["lo_probed32"], lo_u)}]
    assert abs(curve[0]["sw"] - base["strict32"]) < 1e-9, (run, curve[0], base["strict32"])
    for k in range(1, K_MAX + 1):
        c = cuts.get(f"greedy_{TEMP}_k{k}")
        if c is None:
            continue
        sw = strict(c["lo32"], lo_u)
        assert abs(sw - c["strict32"]) < 1e-9, (run, k, sw, c["strict32"])  # my rule == kaudit's stored field
        curve.append({"k": k, "sw": sw})
    rand = []
    sizes = sorted({int(mm.group(1)) for name in cuts if (mm := re.fullmatch(r"rand(\d+)_\d+", name))})
    for n in sizes:  # the figure draws the random sets at the question's cut size only, and not for warm models
        if n != STATUS_K or kind == "warm":
            continue
        vals = [strict(c["lo32"], lo_u) for name, c in cuts.items() if re.fullmatch(rf"rand{n}_\d+", name)]
        if vals:
            rand.append({"k": n, "lo": min(vals), "med": median(vals), "hi": max(vals), "n": len(vals)})
    allh = strict(cuts["all_heads"]["lo32"], lo_u) if "all_heads" in cuts else None
    # reported row (not status): the same greedy audit at a fixed temperature of 16 (a stronger, fixed softening)
    alt = [{"k": 0, "sw": curve[0]["sw"]}] + [{"k": k, "sw": strict(cuts[f"greedy_T16_k{k}"]["lo32"], lo_u)}
                                             for k in range(1, K_MAX + 1) if f"greedy_T16_k{k}" in cuts]
    live = curve[0]["sw"] >= LIVE
    kmin = next((c["k"] for c in curve if c["k"] > 0 and c["sw"] < THRESH), None) if live else None
    return {"run": run, "kind": kind, "seed": int(m.group("s")), "live": live,
            "curve": curve, "random": rand, "kmin": kmin, "all": allh, "alt": alt}


def pct(v):
    return f"{v * 100:.1f}%" if 0 < v < 0.095 else f"{round(v * 100)}%"


def at(r, k):
    return next((c["sw"] for c in r["curve"] if c["k"] == k), None)


def chip(r, gate=None):
    """One line per seed under the panel head. gate: the forced-seed gate note from status.json, if any."""
    who = {"attack": f"seed {r['seed']}", "free": f"seed {r['seed']}", "l0": f"floor, no audit penalty, seed {r['seed']}", "warm": f"before retraining, seed {r['seed']}"}[r["kind"]]
    if r["seed"] == 3 and r["kind"] in ("attack", "free"):
        who += " (spare)"
    if gate and gate.startswith("not counted"):
        s8 = at(r, STATUS_K)
        return f"{who}: {gate}" + (f"; cutting 8 leaves {pct(s8)}" if s8 is not None and at(r, 0) >= LIVE else "")
    if not r["live"]:
        return f"{who}: switch faded ({pct(at(r, 0))} switch with no cut)"
    if r["kind"] not in ("attack", "free"):  # controls: one head already stops an untrained-against switch
        return f"{who}: cutting 1 head leaves {pct(at(r, 1))}"
    s8, own = at(r, STATUS_K), at(r, KATT)
    txt = f"{who}: cutting 6 leaves {pct(own)}, cutting 8 leaves {pct(s8)}"
    if r.get("all") is not None and r["kind"] == "attack":
        txt += f", all 32 heads {pct(r['all'])}"
    return txt + (f" ({gate})" if gate else "")


def build(audit_dir, parent_dir, gates, st=None):
    """Two panels (Tove #31721): 'main' = this run's forced seeds (solid) and the parent's unforced seeds that kept
    the switch (thin dashes) in ONE panel; 'ctl' = a small strip of the controls."""
    mine = [r for r in (load(p) for p in sorted(glob.glob(os.path.join(audit_dir, "*.json")))) if r]
    forced = sorted([r for r in mine if r["kind"] == "attack"], key=lambda r: r["seed"])
    for r in forced:
        r["group"] = "forced"
        r["label"] = f"this run (must keep the switch), seed {r['seed']}"
        if st is not None and r["run"] not in st["counted"]:
            # status.py's frozen gates decide what counts; an uncounted seed is drawn hollow + dotted
            fails = [n.split("failed ", 1)[1] for n in st["notes"] if n.startswith(r["run"] + ":") and "failed " in n]
            r["chip"] = "this run, " + chip(r, f"not counted: failed {fails[0]}" if fails else "not counted")
            r["live"], r["random"] = False, []
            continue
        r["chip"] = "this run, " + chip(r, gates.get(r["run"]))
        # draw the fixed-softening audit only where the frozen audit failed (S(8) >= 0.25)
        if not (r["live"] and (at(r, STATUS_K) or 0) >= THRESH):
            r["alt"] = []
    free, dropped = [], []
    if parent_dir:
        for r in sorted([r for r in (load(p) for p in sorted(glob.glob(os.path.join(parent_dir, "k6_s*.json")))) if r],
                        key=lambda r: r["seed"]):
            if not r["live"]:
                dropped.append(r["seed"]); continue
            r["group"], r["kind"], r["random"], r["alt"] = "free", "free", [], []
            r["label"] = f"previous run (free to drop it), seed {r['seed']}"
            r["chip"] = "previous run, " + chip(r)
            free.append(r)
    main = forced + free
    notes = []
    if dropped:
        notes.append(f"previous run, seeds {' and '.join(map(str, dropped))}: dropped the switch in training (not drawn)")
    panels = [{"key": "main", "name": "Trained against a 6-head audit", "runs": main, "notes": notes}] if main else []
    ctl = sorted([r for r in mine if r["kind"] in ("l0", "warm")], key=lambda r: (r["kind"], r["seed"]))
    for r in ctl:
        r["group"], r["alt"] = "ctl", []
        r["label"] = ("floor, no audit penalty" if r["kind"] == "l0" else "before retraining") + f", seed {r['seed']}"
        r["chip"] = None  # one summary line per kind instead (the strip stays small)
    cnotes = []
    for kind, what in (("l0", "solid: retrained with the floor and no audit penalty"), ("warm", "long dashes: the switch models before retraining")):
        rs = [r for r in ctl if r["kind"] == kind]
        if rs:
            v = [at(r, 1) for r in rs]
            cnotes.append(f"{what} ({len(rs)} seeds): cutting 1 head leaves {pct(min(v))} to {pct(max(v))}")
    if ctl:
        panels.append({"key": "ctl", "small": True, "name": "Controls: no audit penalty in training", "runs": ctl, "notes": cnotes})
    for p in panels:
        assert len({(r["kind"], r["seed"]) for r in p["runs"]}) == len(p["runs"]), f"duplicate seed in {p['key']}"
    return panels


def status(panels):
    p = next((p for p in panels if p["key"] == "main"), None)
    if p is None:
        return "NO DATA", []
    live = [r for r in p["runs"] if r["live"] and r["group"] == "forced"][:3]
    cells = [(r["run"], at(r, STATUS_K), next((x["med"] for x in r["random"] if x["k"] == STATUS_K), None), r["kmin"])
             for r in live]
    if len(live) < 3:
        return "UNCONFIRMED (fewer than 3 live seeds; the forced-seed gate in status.json decides)", cells
    if any(c8 >= THRESH for _, c8, _, _ in cells):
        return "REJECTED", cells
    if all(rm is not None and rm >= RAND_OK for _, _, rm, _ in cells):
        return "CONFIRMED", cells
    return "UNCONFIRMED (random 8-head cut also stops it)", cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default=os.path.join(PROJ, "data", "kaudit"))
    ap.add_argument("--parent-dir", default=None)
    ap.add_argument("--gates", default=None, help="optional JSON {run: short gate note} for the chips")
    ap.add_argument("--status", default=None, help="status.json of status.py: its 'counted' list decides the solid lines")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    gates = json.load(open(a.gates)) if a.gates else {}
    stj = json.load(open(a.status)) if a.status else None
    panels = build(a.audit_dir, a.parent_dir, gates, stj)
    st, cells = status(panels)
    if stj is not None:
        st = f"{stj['status']} (status.py; counted {stj['counted']}; notes {stj['notes']})"
    for p in panels:
        print(f"== {p['key']}: {p['name']}")
        for r in p["runs"]:
            sw = " ".join(f"{c['k']}:{c['sw']:.3f}" for c in r["curve"])
            print(f"{r['run']:10s} live={r['live']!s:5s} kmin={r['kmin']} | {sw}")
            for x in r["random"]:
                print(f"{'':10s} random k={x['k']} n={x['n']} min {x['lo']:.3f} med {x['med']:.3f} max {x['hi']:.3f}")
            print(f"{'':10s} chip: {r['chip']}")
    print("STATUS (figure's own reading; status.py decides):", st, cells)
    if a.check or not a.out:
        return
    data = {
        "_takeaway": "Edit title/subtitle when the numbers land; the builder fills the panels only.",
        "title": "TITLE",
        "subtitle": "SUBTITLE",
        "thresh": THRESH,
        "katt": KATT,
        "status_k": STATUS_K,
        "k_max": K_MAX,
        "panels": [{**{k: v for k, v in p.items() if k != "runs"},
                    "runs": [{k: r[k] for k in ("run", "kind", "group", "seed", "live", "label", "chip", "curve", "random", "kmin", "all", "alt")}
                             for r in p["runs"]]}
                   for p in panels],
    }
    if os.path.exists(a.out):  # keep the hand-written words, refresh the numbers
        old = json.load(open(a.out))
        for k in ("_takeaway", "title", "subtitle"):
            if k in old and not str(old[k]).startswith("PENDING"):
                data[k] = old[k]
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
