"""Main figure data (blog assets/figures/lora-switch-audit/data.json) from data/audit/*.json only.

y = rank of the true cue among 1,001 candidates (itself + 1,000 random cues built the same way) on the stacked
block-26 q,k,v delta's top input direction (PREREG primary e1). x = cue size as a share of what block 26 reads
(x / 608.87, the cue-size-law convention). Reads: 'T' = the snapshot at the switch, '256' = the fixed
behaviour-free checkpoint (C1), 'end' = a parent end-of-run adapter (saved at 1.1-13x T, flagged).
    /usr/bin/python3 figures/figure_data.py [--out <blog>/assets/figures/lora-switch-audit/data.json]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AUD = HERE / "data" / "audit"
DEN = 608.865478515625  # cue-size-law: ||N(h)|| over rows 0-2; share = x / DEN
XSTAR = json.load(open(HERE / "theory" / "xstar.json"))["x_star_pred_PRIMARY_alpha0"][0]


def load(name):
    f = AUD / f"{name}.json"
    return json.load(open(f)) if f.exists() else None


def cell(d, step, read, group, extra=None):
    s = d["snaps"].get(str(step))
    if s is None:
        return None
    q = s["qkv26"]
    x = d["spec"].get("x")
    return {"group": group, "run": d["run"].replace("parent/", ""), "read": read, "step": int(step),
            "x": x, "s": (100 * x / DEN) if x else None, "rank": q["top1"]["rank"], "e1": q["top1"]["true"],
            "p99": q["top1"]["null_p99"], "top4_rank": q["top4"]["rank"], "gain_rank": q["gain"]["rank"],
            "m1": q["top1"]["true"] / q["top1"]["null_p99"], "m4": q["top4"]["true"] / q["top4"]["null_p99"],
            "m1max": q["top1"]["true"] / q["top1"]["null_max"], "m4max": q["top4"]["true"] / q["top4"]["null_max"],
            "mg": q["gain"]["true"] / q["gain"]["null_p99"], "mgmax": q["gain"]["true"] / q["gain"]["null_max"],
            **(extra or {})}


def t_snap(d):
    T = d.get("snap_T") or d.get("T")
    if T is None:
        return None
    sts = sorted(int(k) for k in d["snaps"])
    return min([s for s in sts if s >= T], default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "figures" / "figure_data.json"))
    a = ap.parse_args()
    pts = []
    # C1: six directions at alpha 1/32, read at T and at 256
    for k in range(1110, 1116):
        d = load(f"dir_u{k}_a0.03125_s{k - 1110}")
        if not d:
            continue
        ts = t_snap(d)
        for st, rd in ((ts, "T"), (256, "256")):
            c = cell(d, st, rd, "dirs", {"key": f"u{k}"}) if st else None
            if c:
                pts.append(c)
    # size sweep, u1110 rows 0-2 seed 0, at T
    for f in sorted(AUD.glob("sw_a*_s0.json")):
        d = json.load(open(f))
        ts = t_snap(d)
        if ts:
            pts.append(cell(d, ts, "T", "sweep"))
    # v1.4 length control: the largest cue trained on to the faint cues' switch times (704, 1,728, 4,352)
    for k, sd in ((1110, 0), (1112, 2)):
        d = load(f"len_u{k}_a0.03125_s{sd}")
        if d:
            for st in (256, 704, 1728, 4352):
                c = cell(d, st, "len", "len", {"key": f"u{k}"})
                if c:
                    pts.append(c)
    # smallest learned cue (parent, end-of-run at 1.13-1.26 T)
    for s in (0, 1):
        d = load(f"parent__nocastk8_x0.0876_s{s}")
        if d:
            st = max(int(k) for k in d["snaps"])
            pts.append(cell(d, st, "end", "floor", {"T": d.get("T")}))
    # below the floor: never switched, read at the last snapshot (20,480 = matched to the floor runs)
    for name in ("nocast_x0.0219_s0", "nocast_x0.0055_s0"):
        d = load(name)
        if d and d["snaps"]:
            st = max(int(k) for k in d["snaps"])
            pts.append(cell(d, st, "last", "below", {"final": st >= 20480}))
    # placement: cue on tokens 1-2 only (rows 1-2), cue on token 0 only, the real build leak (parent, end-of-run)
    for f in sorted(AUD.glob("parent__*.json")):
        n = f.stem.replace("parent__", "")
        grp = ("rows12" if ("r12" in n or n.startswith("row12")) else "row0" if n.startswith("row0") else
               "real" if n.startswith("real") else None)
        if grp:
            d = json.load(open(f))
            st = max(int(k) for k in d["snaps"])
            pts.append(cell(d, st, "end", grp, {"T": d.get("T")}))
    # controls: the u1110 cue at alpha 1/32 scored in models that never learned it
    ctrl = []
    for name, lab in (("parent__c2_a0_s0", "no cue"), ("c2_a0_s1", "no cue"), ("c2_a0_s0", "no cue, long"),
                      ("parent__c1_shuf_a0.03125_s0", "cue, reward on a coin"),
                      ("parent__c1_shuf_a0.03125_s1", "cue, reward on a coin"),
                      ("c1_shuf_a0.03125_s0", "cue, reward on a coin")):
        d = load(name)
        if not d or not d["snaps"]:
            continue
        sts = sorted(int(k) for k in d["snaps"] if int(k) > 0)
        for st in sts:
            c = cell(d, st, "ctrl", "ctrl", {"label": lab, "src": "parent" if name.startswith("parent__") else "new"})
            ctrl.append(c)
    # theory band: the B-only limit for the frozen 3-row e1 with the real cue rows and null (theory/ceiling_rows.py,
    # Kasper #20595), union of the 5-95% margin ranges over both output-gradient models and both cue sizes
    cr = json.load(open(HERE / "theory" / "ceiling_rows.json"))
    bands = [v["margin_p5_p95"] for k, v in cr.items() if isinstance(v, dict)]
    data = {
        "theory": {"lo": min(b[0] for b in bands), "hi": max(b[1] for b in bands)},
        "den": DEN, "x_star": XSTAR, "s_star": 100 * XSTAR / DEN,
        "points": [p for p in pts if p and p["s"]],
        "ctrl": ctrl,
        "ctrl_s": 100 * 33.2988 / DEN,
    }
    P = data["points"]
    # C2 primary control reads: new controls at the snapshot nearest the median headline T, parent ones at their end
    hT = sorted(p["step"] for p in P if p["group"] == "dirs" and p["read"] == "T")
    medT = hT[len(hT) // 2] if hT else 32
    prim = []
    for src, run in sorted({(c["src"], c["run"]) for c in ctrl}):
        cs = [c for c in ctrl if c["run"] == run and c["src"] == src]
        old = src == "parent"
        prim.append(max(cs, key=lambda c: c["step"]) if old else min(cs, key=lambda c: abs(c["step"] - medT)))
    data["ctrl_primary"] = prim
    if prim:
        vals = [c["m1"] for c in prim] + [c["mg"] for c in prim]
        data["ctrl_band"] = {"lo": min(vals), "hi": max(vals)}
    learned = [p for p in P if p["group"] in ("sweep", "floor") or (p["group"] == "dirs" and p["read"] == "T")]
    n4 = sum(p["gain_rank"] == 1 for p in learned)
    big = [p for p in P if p["group"] == "dirs" and p["read"] == "256"]
    nb = sum(p["rank"] == 1 for p in big)
    # largest share s such that every learned-cue point at or below s ranks first on the top direction
    lp = sorted([p for p in P if p["group"] in ("sweep", "floor")], key=lambda p: p["s"])
    s_flip = None
    for p in lp:
        if p["rank"] != 1:
            break
        s_flip = p["s"]
    pctx = lambda v: f"{v:.2g}%"  # noqa: E731
    data["s_flip"] = s_flip
    ln = [p for p in P if p["group"] == "len" and p["step"] == 4352]
    nl = sum(p["rank"] == 1 for p in ln)
    data.update({
        "title": (f"The top direction of the weight change missed the largest cue in {len(big) - nb} of {len(big)} "
                  f"models at 256 updates; trained as long as the faintest cues, it found it in {nl} of {len(ln)}"),
        "subtitle": ("Two weight-only scores of the true cue, each divided by the level that 1 in 100 of 1,000 random "
                     "cues reach. Above 1x, the score singles the cue out. The cue shrinks to the right."),
        "x_label": "Cue size, as a share of what the trained layer reads (log)",
        "x_short": "Cue size, share of layer input (log)",
        "y_label": "True cue vs random cues (log)",
        "y_short": "True cue vs random (log)",
        "legend": {"top1": "filled: top direction of the change", "top4": "hollow: reaction of the whole change to the cue",
                   "sweep": "circles: one run per cue size, read at the switch (number: updates)",
                   "big": "squares: largest cue, six models, 256 updates", "real": "diamonds: real build difference, 2 runs",
                   "below": "dashed diamond: 4x below the smallest learned cue, never switched",
                   "len": "arrows: largest cue, trained on to 704, 1,728 and 4,352 updates",
                   "ctrl": "grey band: controls, cue never rewarded", "theory": "teal band: theory, change carries only the cue"},
        "notes": {"p99": "1 in 100 random cues reach this line", "len": "trained on (updates)", "star": "predicted crossover (exploratory)",
                  "ctrl": "controls"},
    })
    # v2 layout (Tove #20672): x = training updates. One line per run over all its snapshots.
    def series(name, kind, label=None):
        d = load(name)
        if not d:
            return None
        pts = []
        for st in sorted(int(k) for k in d["snaps"]):
            if st == 0:
                continue
            q = d["snaps"][str(st)]["qkv26"]
            pts.append({"step": st, "m1": q["top1"]["true"] / q["top1"]["null_p99"], "rank": q["top1"]["rank"],
                        "mg": q["gain"]["true"] / q["gain"]["null_p99"], "gain_rank": q["gain"]["rank"]})
        x = d["spec"].get("x")
        return {"run": d["run"].replace("parent/", ""), "kind": kind, "x": x, "s": 100 * x / DEN if x else None,
                "T": d.get("snap_T") or d.get("T"), "label": label, "pts": pts}
    S = []
    for f in sorted(AUD.glob("sw_a*_s0.json")):
        S.append(series(f.stem, "sweep"))
    for k, sd in ((1110, 0), (1112, 2)):
        S.append(series(f"len_u{k}_a0.03125_s{sd}", "sweep"))
    for sd in (0, 1):
        S.append(series(f"parent__nocastk8_x0.0876_s{sd}", "floor"))
    S.append(series("nocast_x0.0219_s0", "below"))
    S.append(series("c2_a0_s0", "ctrl", "no cue"))
    for sd in (0, 1):
        S.append(series(f"parent__real_s{sd}", "real"))
    data["series"] = [x for x in S if x and x["pts"]]
    # first update at which each learned run's top-direction score reached the 1-in-100 level
    first = {}
    for r in data["series"]:
        if r["kind"] in ("sweep", "floor"):
            hit = [p["step"] for p in r["pts"] if p["m1"] >= 1]
            first[r["run"]] = hit[0] if hit else None
    data["first_hit"] = first
    past = [r for r in data["series"] if r["kind"] in ("sweep", "floor") and r["pts"][-1]["step"] >= 1024]
    allpast = all(any(p["m1"] >= 1 for p in r["pts"] if p["step"] >= 1024) for r in past)
    data["title"] = (f"At 256 updates the top direction missed the largest cue in {len(big) - nb} of {len(big)} models; "
                     + (f"every learned random cue trained past 1,000 updates reached it ({len(past)} of {len(past)} runs)"
                        if allpast else "some learned random cues trained past 1,000 updates still missed it"))
    data["subtitle"] = ("A weight-only score of the true cue on the top direction of the reward-trained weight change, "
                        "divided by the level that 1 in 100 of 1,000 random cues reach, over training. "
                        "Above 1x, the score singles the cue out.")
    data["x2_label"] = "Training updates (log)"
    data["notes"]["below"] = "never learned"
    data["legend"].update({
        "line": "one run; colour = cue size (share of what the layer reads)",
        "switch": "large dot: the update where the switch appeared",
        "big2": "squares: largest cue, six models at 256 updates",
        "below2": "dashed: cue 4x below the smallest learned one, never learned",
        "ctrl2": "thin line: no cue in training (control)",
        "real2": "diamonds: real build leak, 2 runs",
        "floor": "triangles: smallest learned cue, 2 runs",
        "gain2": "hollow: reaction of the whole change, at each run's last read",
        "theory": "teal band: theory, change carries only the cue",
        "ctrl": "lower shaded band: controls, cue not tied to reward"})
    # text first: readers of the data head (oracle figure_alone gate reads the first 3,000 chars) see every label
    front = ["title", "subtitle", "x2_label", "y_label", "y_short", "legend", "notes"]
    data = {**{k: data[k] for k in front if k in data}, **{k: v for k, v in data.items() if k not in front}}
    # drop v1 legend keys the plot no longer draws
    for k in ("top1", "top4", "sweep", "big", "real", "below", "len"):
        data["legend"].pop(k, None)
    for k in ("star", "len", "ctrl"):
        data["notes"].pop(k, None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1))
    for p in data["points"]:
        print(p["group"], p["run"], p["read"], p["step"], round(p["s"], 4), "rank", p["rank"], "gain_rank", p["gain_rank"], "m1 %.2f mg %.2f" % (p["m1"], p["mg"]))
    print("controls", [(c["run"], c["step"], c["rank"], round(c["m1"], 2), round(c["m4"], 2)) for c in ctrl])


if __name__ == "__main__":
    main()
