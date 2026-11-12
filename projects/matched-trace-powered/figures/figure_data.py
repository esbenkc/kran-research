"""Main figure data for matched-trace-powered (owner Owain). Reads data/summary.json (summarize.py, the one source for
every number) and writes blog.kran.ai/assets/figures/matched-trace-powered/data.json.

One training run, two cues in it (the 11-11 run, updates 0-674, continued here to the clock stop).
x = training thoughts in which a cue gave the probe away: text difference seen in training, summed over probed thoughts (planted cue: cum_P_tv_mass, exact
    0.10 per probed thought; probe's trace: cum_R_tv_mass, trapezoid of dev D_R). Global over both nights.
y = share of the limit: switch / (text difference x cheating with no probe). 0% = the model ignores the cue.
  planted cue, dev checks: dev forced switch / dev cheat_none (every 50 updates, 64 dev puzzles).
  The probe's trace has no plotted dev checks (64 dev puzzles x 1 draw: share noise about +-0.5).
  Full TEST reads with 95% intervals, both cues: start adapter, the 11-11 end adapter (update 674), the end adapter.
Shapes: the frozen predicted planted share band at the end mass (PREREG v1), the trace's 'not learned' margin
(share CI-high < 10%, PREREG status rule).

    python3 figures/figure_data.py            # from data/summary.json
    python3 figures/figure_data.py --mock     # layout test: 11-11 numbers + made-up continuation
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "matched-trace-powered"
MARGIN = 0.10  # PREREG status rule: trace share CI-high below this = not learned
# PREREG.md 'Frozen predicted number for the planted cue' (Iris #19849, before T0): centre by end cue mass m,
# linear between table points and beyond them; 80% band = centre x [0.33, 1.67], capped to [0, 1].
BAND_M = [3000, 3200, 3450, 3650, 3900]
BAND_C = [0.19, 0.27, 0.45, 0.60, 0.76]


def band_at(m):
    import re
    txt = (HERE / "PREREG.md").read_text()
    row_m = re.search(r"\| end cue mass m \|(.*)\|", txt).group(1)
    row_c = re.search(r"\| central s_P \|(.*)\|", txt).group(1)
    assert [float(v.replace(",", "")) for v in row_m.split("|")] == BAND_M
    assert [float(v) for v in row_c.split("|")] == BAND_C
    k = 0 if m < BAND_M[1] else (len(BAND_M) - 2 if m >= BAND_M[-2] else max(i for i in range(len(BAND_M) - 1) if BAND_M[i] <= m))
    c = BAND_C[k] + (BAND_C[k + 1] - BAND_C[k]) * (m - BAND_M[k]) / (BAND_M[k + 1] - BAND_M[k])
    return {"x": m, "c": c, "lo": max(0.0, 0.33 * c), "hi": min(1.0, 1.67 * c)}


def end_pts(cell, xP, xR, when, kP="s_P_rb", kR="s_R"):
    """Two TEST-read points (planted, real) from one read cell of summary.json."""
    if not cell:
        return []
    out = []
    for cue, x, k in (("planted", xP, kP), ("real", xR, kR)):
        c = cell.get(k)
        if not c or x is None:
            continue
        out.append({"cue": cue, "x": x, "s": c["v"], "lo": c["ci"][0], "hi": c["ci"][1], "when": when,
                    "D": cell["D_P" if cue == "planted" else "D_R"]["v"]})
    return out


def from_summary():
    s = json.load(open(HERE / "data" / "summary.json"))
    F = s["figure"]  # written by summarize_tf.py for this figure: global checks, read cells and their x
    pts = [{"cue": "planted", "step": r["step"], "x": r["x"], "y": r["s"], "forced": r.get("forced")} for r in F["checks"]]
    ends = []
    for cell in F["reads"]:
        # Kasper #19891: the sharper (likelihood-ratio) trace read was chosen after the 11-11 result, so the start and
        # update-674 trace dots use the plain read frozen then; only the end read uses the pre-registered sharper one.
        kR = cell.get("kR", "s_R") if cell["when"] == "final" else "s_R_plain"
        ends += end_pts(s[cell["key"]], cell["xP"], cell["xR"], cell["when"], cell.get("kP", "s_P"), kR)
    fin = [c for c in F["reads"] if c["when"] == "final"]
    band = None
    if fin:
        band = band_at(fin[0]["xP"])
        if F.get("band"):  # summarize_tf.py's own band (Iris's centre function) must agree with the PREREG table
            assert abs(F["band"]["central"] - band["c"]) < 0.01, (F["band"], band)
    end = s[fin[0]["key"]] if fin else {}
    return pts, F.get("onset"), band, ends, {"status": s.get("status"), "steps": F.get("steps"),
                                             "diff": end.get(F.get("diff_key", "gap")), "night1_x": F.get("night1_x")}


def mock():
    """11-11 checks and reads as they are, plus a made-up continuation to ~1,100 updates (layout only)."""
    p = json.load(open(HERE / "data" / "parent" / "summary_1111.json"))
    pts = [{"cue": "planted", "step": r["step"], "x": r["cum_P_tv_mass"] or 0.0, "y": r["dev_forced_switch"] / r["cheat_none"],
            "forced": r["dev_forced_switch"]} for r in p["curve"]["checks"] if r.get("dev_forced_switch") is not None]
    for st in range(700, 1101, 50):
        x = 3.2 * st
        pts.append({"cue": "planted", "step": st, "x": x, "y": 0.1 + 0.25 * (x - 2157) / 1300, "forced": None})
    ends = end_pts(p["start"], 0.0, 0.0, "start") + end_pts(p["final"], p["cue_tv_mass_final"], p["R_tv_mass_final"], "674")
    ends += [{"cue": "planted", "x": 3520, "s": 0.31, "lo": 0.25, "hi": 0.37, "when": "final", "D": 0.1},
             {"cue": "real", "x": 3300, "s": 0.02, "lo": -0.05, "hi": 0.08, "when": "final", "D": 0.095}]
    on = p["curve"]["onset"]
    return pts, {"step": on["step"], "x": on["cue_tv_mass"]}, {"x": 3520, "lo": 0.2, "hi": 0.45}, ends, \
        {"status": "MOCK", "steps": 1100, "diff": None, "night1_x": p["cue_tv_mass_final"]}


def _pc(v):  # one decimal, as in the post's TLDR
    sign = "\u2212" if v < -0.00005 else ""
    return f"{sign}{abs(v) * 100:.1f}%"


def head(meta, ends):
    """takeaway title (how the result looks; never the post's question or claim title)"""
    if meta.get("status") == "MOCK":
        return "MOCK HEAD"
    return "Through training, the planted cue kept a small switch. The probe's trace showed none"


def answer(meta, ends):
    fin = {e["cue"]: e for e in ends if e.get("when") == "final"}
    if len(fin) < 2:
        return ""
    p, r = fin["planted"], fin["real"]
    lead = {"CONFIRMED": "Yes.", "REJECTED": "No.", "UNCONFIRMED": "Not settled."}.get(meta.get("status"), "")
    return (f"{lead} At the end: planted cue {_pc(p['s'])} of its limit ({_pc(p['lo'])} to {_pc(p['hi'])}); "
            f"probe's trace {_pc(r['s'])} ({_pc(r['lo'])} to {_pc(r['hi'])}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    pts, onset, band, ends, meta = mock() if a.mock else from_summary()
    data = {
        "mock": bool(a.mock),
        "title": head(meta, ends),
        "subtitle": ("Qwen3-0.6B, one training run of " + f"{meta.get('steps') or 0:,}" + " updates. Share of the limit: the switch "
                     "(how much less the model cheats when the cue is in its thought) divided by the switch of a model that "
                     "stops cheating whenever the words give the probe away. Both cues change the thought by about 10 points "
                     "of text difference."),
        "x_label": "Training thoughts in which a cue gave the probe away, expected count (thousands) \u2192",
        "x_label_short": "Thoughts that gave the probe away, expected (thousands) \u2192",
        "y_label": "\u2191 Share of the limit",
        "legend": {"planted": "planted cue: checks during training (64 puzzles; in training \u201cNeed\u201d is made more likely, in these checks it is written in)",
                   "real": "probe's trace: full tests only (hollow dot, 95% interval)",
                   "band": "planted cue at the end, as predicted before the run",
                   "margin": "the trace counts as showing no switch if its whole interval sits below this line",
                   "end": "full test, 95% interval"},
        "band_label": "predicted", "margin": MARGIN, "band": band, "night1_x": meta.get("night1_x"), "onset": onset, "points": pts, "ends": ends, "meta": meta,
        "notes": ["Hollow dots: full tests on the 200 test puzzles, before training, after the first night and at the end. "
                  "The trace's first two tests compare sampled probe-on and probe-off thoughts. The end test reweights each probe-off thought by how likely the probe would make it, so its interval is narrower."],
        "answer": answer(meta, ends),
    }
    BLOG.mkdir(parents=True, exist_ok=True)
    (BLOG / "data.json").write_text(json.dumps(data, indent=1))
    print("wrote", BLOG / "data.json", len(pts), "dev points,", len(ends), "test points", meta)


if __name__ == "__main__":
    main()
