"""Build the main figure's data.json (blog assets/figures/reader-bound-shield/) from the raw reads and reader fits.

The gate failed at update 200 and D3 stopped it there (DEVIATIONS D3), so the real probe was never trained and the
figure has one row per trained model (plus the planted cue before training), not a y-vs-x scatter: only the
update-100 checkpoint has a reader fitted on its own thoughts (DEVIATIONS D4).
Per row, all in points (x100):
  D = text difference of the read that gives the switch (read.py 'D', total variation between probe-on and probe-off
      thoughts; DEVIATIONS D1): the ceiling on the switch and on any reader edge;
  C = reader edge (reader_sup2.py 'x', 'x_ci', frozen reader rule), when a reader was fitted on these thoughts;
  S = served switch: mean over test CRN pairs of cheat(probe off) - cheat(probe on) (read.py 'gap', 'gap_ci');
  bar = the bar that row's switch had to pass, fixed before it ran (gate 5 points; dense diagnostic 10 points).
Run from the project folder: python3 figures/figure_data.py [--blog ../../../blog.kran.ai]. CPU, seconds, stdlib only.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

ROWS = [
    dict(name="start", kind="planted", short="planted cue, before training", label="planted cue ' Need', before training",
         read="data/parent/start_planted_test.json", reader="research-iris/x_planted_start.json", trained=False),
    dict(name="p100", kind="planted", short="planted cue, update 100", run="planted_s0", step=100, label="planted cue ' Need', after 100 updates",
         read="data/reads/raw/planted_s0_gate_100.json", reader="research-iris/x_planted_end.json", bar=0.05, trained=True),
    dict(name="p200", kind="planted", short="planted cue, update 200", run="planted_s0", step=200, label="planted cue ' Need', after 200 updates",
         read="data/reads/raw/planted_s0_gate_200.json", reader=None, bar=0.05, trained=True),
    dict(name="dense", kind="dense", short="dense cue, update 200", run="dense_s0", step=200,
         label="dense cue: ' Need' first on nearly every probe-on thought, after 200 updates",
         read="data/reads/raw/dense_s0_gate_200.json", reader=None, bar=0.10, trained=True),
]


def iv(v, ci):
    return dict(v=v, lo=ci[0], hi=ci[1]) if ci else dict(v=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=os.path.join(PROJ, "../../../blog.kran.ai"))
    a = ap.parse_args()
    out = os.path.join(a.blog, "assets/figures/reader-bound-shield/data.json")
    base = json.load(open(out))
    summ = json.load(open(os.path.join(PROJ, "data/summary.json"))).get("cells", {})
    by_file = {c.get("file"): c for c in summ.values()}
    expo = json.load(open(os.path.join(PROJ, "data/summary.json"))).get("exposure", {})
    rows, missing = [], []
    for spec in ROWS:
        rp = os.path.join(PROJ, spec["read"])
        if not os.path.exists(rp):
            missing.append(spec["name"])
            continue
        rd = json.load(open(rp))["test"]
        row = dict(name=spec["name"], kind=spec["kind"], short=spec["short"], label=spec["label"],
                   D=iv(rd["D"], rd.get("D_ci")), f=rd.get("f"), read=spec["read"])
        if spec.get("run"):
            m = next((e["marked"] for e in expo.get(spec["run"], []) if e["step"] == spec["step"]), None)
            if m is not None:
                row["marked"] = m
                row["short"] = f"{spec['short']} (~{int(round(m, -2)):,} marked)"
        if spec.get("bar") is not None:
            row["bar"] = spec["bar"]
        if spec["trained"]:
            row["S"] = iv(rd["gap"], rd["gap_ci"])
            if rd.get("forced_switch") is not None:
                row["F"] = iv(rd["forced_switch"], rd.get("forced_switch_ci"))
        cs = by_file.get(spec["read"], {}).get("ceiling_share")
        if spec["trained"] and cs:
            row["share"] = dict(v=cs["share"], lo=cs["ci"][0], hi=cs["ci"][1])
        cp = spec["reader"] and os.path.join(PROJ, spec["reader"])
        if cp and os.path.exists(cp):
            c = json.load(open(cp))
            row["C"] = iv(c["x"], c["x_ci"])
            row["C_note"] = f"{c.get('stop')} after {c.get('n_fit_pairs')} labelled pairs" + (
                ", a lower bound" if c.get("stop") == "capped" else "")
            row["reader"] = spec["reader"]
        rows.append(row)
    if rows:
        base["rows"] = rows
        base["missing"] = missing
        base["mock"] = False
    json.dump(base, open(out, "w"), indent=1)
    print(f"{len(rows)} rows -> {out}; missing {missing}")
    fmt = lambda d: "-" if not d else "%6.2f [%.2f, %.2f]" % (d["v"] * 100, d["lo"] * 100, d["hi"] * 100)
    for r in rows:
        print(f"  {r['short']:26s} D {r['D']['v'] * 100:6.2f}  C {fmt(r.get('C'))}  S {fmt(r.get('S'))}  {r['read']}")

if __name__ == "__main__":
    main()
