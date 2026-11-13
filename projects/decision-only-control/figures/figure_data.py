"""Main figure data (Owain): data/summary.json (summ.py, Bruno) -> blog.kran.ai/assets/figures/decision-only-control/data.json.

Series = one decision-only run each (no-std recipe), y = s_P (switch as a share of its limit) at every read; the probe's
trace = s_R of the 0.10 run (the only run whose trace counts: in the others the planted word already marks many probed
thoughts). The last read of the 0.10 run and of its trace carry 95% intervals (prompt-clustered bootstrap, summ.py).
Bands = the predicted onset of the 0.10 run from T86, frozen in the room before the 0.10 run started (summary 'slope').
Text (title, subtitle, answer, notes) lives in figures/main_text.json; @@key@@ is filled from the numbers below.

    /usr/bin/python3 figures/figure_data.py [--mock]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "decision-only-control" / "data.json"
RUNGS = [("d086_s0", "rung"), ("d030_s0", "rung"), ("d010_s0", "hero")]


def num(v):
    return f"{v:,.0f}"


def main():
    S = json.loads((HERE / "data" / "summary.json").read_text())
    T = json.loads((HERE / "figures" / "main_text.json").read_text())
    runs, sl = S["runs"], S["slope"]
    series, flat = [], {}
    for key, kind in RUNGS:
        if key not in runs:
            continue
        rd = runs[key]["reads"]
        s = {"key": key, "kind": kind, "label": T["labels"][key], "short": T["short"][key],
             "pts": [{"u": r["step"], "y": round(r["s_P"]["v"], 4)} for r in rd]}
        e = rd[-1]
        flat[f"{key}.end_u"] = num(e["step"])
        flat[f"{key}.end_s"] = f'{100 * e["s_P"]["v"]:.0f}%'
        flat[f"{key}.end_ci"] = f'{100 * e["s_P"]["ci"][0]:.0f}% to {100 * e["s_P"]["ci"][1]:.0f}%'
        if kind == "hero":
            s["end"] = {"u": e["step"], "y": round(e["s_P"]["v"], 4), "lo": round(e["s_P"]["ci"][0], 4), "hi": round(e["s_P"]["ci"][1], 4)}
            tr = {"key": "trace", "kind": "trace", "label": T["labels"]["trace"], "short": T["short"]["trace"],
                  "pts": [{"u": r["step"], "y": round(r["s_R"]["v"], 4)} for r in rd],
                  "end": {"u": e["step"], "y": round(e["s_R"]["v"], 4), "lo": round(e["s_R"]["ci"][0], 4), "hi": round(e["s_R"]["ci"][1], 4)}}
            flat["trace.end_s"] = f'{100 * e["s_R"]["v"]:.0f}%'.replace("-", "\u2212")
            flat["trace.end_ci"] = f'{100 * e["s_R"]["ci"][0]:.0f}% to {100 * e["s_R"]["ci"][1]:.0f}%'.replace("-", "\u2212")
            flat["start_s"] = f'{100 * rd[0]["s_P"]["v"]:.1f}%'
        series.append(s)
    if "d010_s0" in runs:
        series.append(tr)
    lo_d = sl["predicted_T10_drift"] * sl["T86_interval"][0] / sl["T86"]
    lo_n = sl["predicted_T10_noise"] * sl["T86_interval"][0] / sl["T86"]
    flat.update({"T86": num(sl["T86"]), "drift": num(sl["predicted_T10_drift"]), "noise": num(sl["predicted_T10_noise"]),
                 "drift_lo": num(lo_d), "noise_lo": num(lo_n)})

    def fill(x):
        out = re.sub(r"@@([\w.]+)@@", lambda m: flat[m.group(1)], x)
        assert "@@" not in out, out
        return out
    T = json.loads(fill(json.dumps(T)))
    data = {"mock": False, "title": T["title"], "subtitle": T["subtitle"], "x_label": T["x_label"], "y_label": T["y_label"],
            "legend": T["legend"],
            "bands": [{"x0": round(lo_d), "x1": round(sl["predicted_T10_drift"]), "label": T["band_drift"]},
                      {"x0": round(lo_n), "x1": round(sl["predicted_T10_noise"]), "label": T["band_noise"]}],
            "bar": {"y": 0.30}, "xmax": T.get("xmax"), "series": series, "notes": T["notes"], "answer": T["answer"],
            "source": "kran-research projects/decision-only-control/data/summary.json (summ.py); figures/figure_data.py"}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT, {k: v for k, v in flat.items()})


if __name__ == "__main__":
    sys.exit(main())
