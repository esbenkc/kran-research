"""Main figure data for matched-trace (owner Felix, T8). Reads data/summary.json (written by summarize.py, the one
source for every number) and writes blog.kran.ai/assets/figures/matched-trace/data.json.

x = text difference seen in training, summed over probed thoughts (summarize.py): planted cue = cum_P_tv_mass (exact,
    0.10 x probed rows); probe's own trace = cum_R_tv_mass (trapezoid of dev D_R on the 50-update grid x 32 rows).
y = switch as a share of its ceiling, s = y / (D x cheat_none) (PREREG 'Reads').
  planted cue, dev checks (DEVIATIONS D1): s = dev forced switch / dev cheat_none (precise: every dev row counts).
  The real trace's dev checks are NOT plotted: 64 dev puzzles x 1 draw give a share noise of about +-0.5.
  Both cues get two full TEST reads with 95% intervals: the start adapter (x = 0) and the final adapter.

    python3 figures/figure_data.py            # from data/summary.json
    python3 figures/figure_data.py --mock     # layout test
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "matched-trace"
BAND = [1359, 3198]  # PREREG v1 frozen planted-onset band, in cue text-difference mass


def end_pts(cell, xP, xR, key_P):
    """Two TEST-read points (planted, real) from one read cell of summary.json."""
    if not cell:
        return []
    out = []
    for cue, x, k in (("planted", xP, key_P), ("real", xR, "s_R")):
        c = cell[k]
        yk = "y_P_rb" if (cue == "planted" and k == "s_P_rb") else ("y_P" if cue == "planted" else "y_R")
        out.append({"cue": cue, "x": x, "s": c["v"], "lo": c["ci"][0], "hi": c["ci"][1],
                    "y": cell[yk]["v"], "ylo": cell[yk]["ci"][0], "yhi": cell[yk]["ci"][1],
                    "D": cell["D_P" if cue == "planted" else "D_R"]["v"]})
    return out


def from_summary():
    s = json.load(open(HERE / "data" / "summary.json"))
    cv = s["curve"]
    pts = [{"cue": "planted", "step": r["step"], "x": r["cum_P_tv_mass"] or 0.0,
            "y": r["dev_forced_switch"] / r["cheat_none"], "forced": r["dev_forced_switch"]}
           for r in cv["checks"] if r.get("dev_forced_switch") is not None and r.get("cheat_none")]
    on = cv.get("onset")
    onset = {"step": on["step"], "x": on["cue_tv_mass"]} if on else None
    key_P = s.get("primary_P", {}).get("s_P_key", "s_P_rb")
    ends = end_pts(s.get("start"), 0.0, 0.0, key_P) + end_pts(s.get("final"), s.get("cue_tv_mass_final"),
                                                              s.get("R_tv_mass_final"), key_P)
    for e in ends:
        e["when"] = "start" if e["x"] == 0.0 else "final"
    diff = s.get("final", {}).get(s.get("primary_P", {}).get("diff_key", "s_P_rb_minus_s_R"))
    steps = cv.get("steps_done") or (cv["checks"][-1]["step"] if cv["checks"] else 0)
    rm = s.get("R_tv_mass_final") or (cv["checks"][-1].get("cum_R_tv_mass") if cv["checks"] else None)
    d_avg = rm / (32 * steps) if (rm and steps) else None  # mean text difference of the trace over training
    return pts, onset, ends, {"status": s.get("status"), "diff": diff, "D_R_avg": d_avg,
                              "ratio_R": s.get("exposure_ratio_R_final_over_P_onset")}


def mock():
    import math
    pts = [{"cue": "planted", "step": st, "x": 3.2 * st, "y": 0.25 / (1 + math.exp(-(3.2 * st - 900) / 200)),
            "forced": 0.1} for st in range(0, 700, 50)]
    ends = [{"cue": "planted", "x": 0, "s": -0.02, "lo": -0.06, "hi": 0.02, "y": -0.001, "ylo": -0.003, "yhi": 0.001, "when": "start", "D": 0.1},
            {"cue": "real", "x": 0, "s": -0.05, "lo": -0.2, "hi": 0.1, "y": -0.003, "ylo": -0.01, "yhi": 0.006, "when": "start", "D": 0.13},
            {"cue": "planted", "x": 2200, "s": 0.24, "lo": 0.19, "hi": 0.29, "y": 0.017, "ylo": 0.013, "yhi": 0.02, "when": "final", "D": 0.1},
            {"cue": "real", "x": 1900, "s": 0.01, "lo": -0.13, "hi": 0.15, "y": 0.001, "ylo": -0.008, "yhi": 0.01, "when": "final", "D": 0.08}]
    return pts, {"step": 200, "x": 640}, ends, {"status": "MOCK", "D_R_avg": 0.09}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    pts, onset, ends, meta = mock() if a.mock else from_summary()
    data = {
        "mock": bool(a.mock),
        "title": "TITLE_SET_FROM_RESULT",
        "subtitle": "One training run of Qwen3-0.6B. The planted cue changes 1 sampled thought in 10"
                    + (f"; the probe's trace changed about 1 in {1 / meta['D_R_avg']:.0f} on average." if meta.get("D_R_avg")
                       else "."),
        "x_label": "Thoughts the hint changed during training (thousands, expected count) \u2192",
        "x_label_short": "Thoughts the hint changed (thousands) \u2192",
        "y_label": "\u2191 Switch, as % of what the text allows",
        "legend": {"planted": "planted cue (the word \u201cNeed\u201d made more likely)",
                   "real": "the probe's own trace (its rounding changes words)",
                   "band": "predicted start of learning for the planted cue",
                   "onset": "planted cue starts to be learned",
                   "end": "full test on new puzzles (95% interval)"},
        "band": BAND, "onset": onset, "points": pts, "ends": ends, "meta": meta,
        "notes": ["Small dots: checks during training (planted cue only; the trace's checks are too noisy to plot)."],
        "ratio_label": (f"saw {meta['ratio_R']:.1f}x what the planted cue needed to start" if meta.get("ratio_R") else
                        ("saw 3.0x what the planted cue needed to start" if a.mock else None)),
        "answer": "ANSWER_SET_FROM_RESULT",
    }
    BLOG.mkdir(parents=True, exist_ok=True)
    (BLOG / "data.json").write_text(json.dumps(data, indent=1))
    print("wrote", BLOG / "data.json", len(pts), "dev points,", len(ends), "test points", meta)


if __name__ == "__main__":
    main()
