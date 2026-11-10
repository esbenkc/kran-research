"""Build the main figure's data.json (blog assets/figures/reader-bound-shield/) from the raw reads and reader fits.

One point per trace (planted cue, real probe, scattered cue, bit-exact probe):
  y = served switch: mean over test CRN pairs of cheat(probe off) - cheat(probe on), with its prompt-bootstrap CI
      (read.py 'test' block: gap, gap_ci);
  x = reader edge: the frozen reader rule's x on the same trace's start texts (reader_sup2.py: x, x_ci; for an arm
      with an end-text fit, the LARGER of start and end x, PREREG 'Reader rule');
  D = the text difference of the read that gives y (read.py 'D', total variation between probe-on and probe-off
      thoughts): the ceiling on |y| and |x| for any coupling (DEVIATIONS D1); f (share of differing pairs) is kept for
      the tooltip only.
Run from the project folder: python3 figures/figure_data.py [--blog ../../../blog.kran.ai]. CPU, seconds, stdlib only.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

# trace -> (served read json, start-text reader json(s), end-text reader json or None, labels)
TRACES = [
    dict(kind="planted", short="planted cue", label="planted cue (one token), seed 0",
         read="data/reads/raw/planted_s0_gate_*.json", reader="research-iris/x_planted_start*.json", end=None),
    dict(kind="real", short="real probe", label="real probe, seed 0",
         read="data/reads/raw/real_s0.json", reader="research-iris/x_real_start*.json", end=None),
    dict(kind="hash", short="scattered cue", label="scattered cue, seed 0",
         read="data/reads/raw/hash_s0.json", reader="research-iris/x_hash_start*.json", end=None),
    dict(kind="real", short="real probe s1", label="real probe, seed 1",
         read="data/reads/raw/real_s1.json", reader="research-iris/x_real_start*.json", end=None),
    dict(kind="exact", short="bit-exact probe", label="bit-exact probe (trained model of the real probe)",
         read="data/reads/raw/exact_*.json", reader=None, end=None),
]


def one(pattern, pick_last=True):
    fs = sorted(glob.glob(os.path.join(PROJ, pattern)), key=os.path.getmtime)
    return fs[-1] if fs else None


def gate_pass_read():
    """The planted point is the read that passed the gate (gate.jsonl pass=True), else the last gate read."""
    g = os.path.join(PROJ, "data/runs/planted_s0/gate.jsonl")
    if not os.path.exists(g):
        return None
    rows = [json.loads(l) for l in open(g) if l.strip()]
    if not rows:
        return None
    r = next((r for r in rows if r.get("pass")), rows[-1])
    return os.path.join(PROJ, f"data/reads/raw/planted_s0_gate_{r['step']}.json")


def reader_x(pattern):
    """Full-data (frac 1.0) fit of the frozen rule; returns (x, lo, hi, stop, n_fit_pairs, path)."""
    if not pattern:
        return None
    fs = [f for f in glob.glob(os.path.join(PROJ, pattern)) if not f.endswith("_pairs.npz")]
    best = None
    for f in sorted(fs, key=os.path.getmtime):
        d = json.load(open(f))
        if "x" not in d or float(d.get("frac", 1.0)) != 1.0 or d.get("null") not in (None, "none"):
            continue
        best = (d["x"], d["x_ci"][0], d["x_ci"][1], d.get("stop"), d.get("n_fit_pairs"), f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=os.path.join(PROJ, "../../../blog.kran.ai"))
    a = ap.parse_args()
    out = os.path.join(a.blog, "assets/figures/reader-bound-shield/data.json")
    base = json.load(open(out))
    pts, notes = [], []
    for t in TRACES:
        rp = gate_pass_read() if t["kind"] == "planted" else one(t["read"])
        if not rp or not os.path.exists(rp):
            continue
        rd = json.load(open(rp))["test"]
        y = dict(v=rd["gap"], lo=rd["gap_ci"][0], hi=rd["gap_ci"][1])
        if t["kind"] == "exact":
            x = dict(v=0.0, lo=0.0, hi=0.0)
            xn = "zero by construction: the two thoughts are identical"
        else:
            rx = reader_x(t["reader"])
            if not rx:
                continue
            x = dict(v=rx[0], lo=rx[1], hi=rx[2])
            xn = f"{rx[3]} after {rx[4]} labelled pairs" + (", a lower bound" if rx[3] == "capped" else "")
        pts.append(dict(kind=t["kind"], short=t["short"], label=t["label"], x=x, y=y, x_note=xn,
                        D=rd.get("D", 0.0), D_ci=rd.get("D_ci"), f=rd.get("f"), read=os.path.relpath(rp, PROJ)))
    if pts:
        base["points"] = pts
        base["mock"] = False
        base["notes"] = notes or base.get("notes", [])
    json.dump(base, open(out, "w"), indent=1)
    print(f"{len(pts)} points -> {out}" + ("" if pts else " (no data yet: kept the mock)"))
    for p in pts:
        print(f"  {p['short']:16s} x {p['x']['v'] * 100:6.2f} [{p['x']['lo'] * 100:6.2f}, {p['x']['hi'] * 100:6.2f}]  "
              f"y {p['y']['v'] * 100:6.2f} [{p['y']['lo'] * 100:6.2f}, {p['y']['hi'] * 100:6.2f}]  D {p['D']}  f {p['f']}  {p['read']}")


if __name__ == "__main__":
    main()
