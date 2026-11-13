"""Build data.json for the main figure of on-policy-planted (Selma, T5).

Input: figures/data.json (live_summ.py, from raw reads): live planted ' We' and probe trace (this run), cached
planted ' We' (decision-only-control d010_s0, thoughts sampled once), onset intervals. Band: the frozen 2x window (300, 1,200] around the cached
onset (PREREG.md 'The frozen predicted number'). End dots: the last live read with its 95% intervals.

    /usr/bin/python3 figures/figure_data.py
Writes figures/main_figure.json (live_summ.py owns figures/data.json) and copies it to the blog clone's assets/figures/on-policy-planted/data.json when present.
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "on-policy-planted"


src = json.loads((HERE / "figures/data.json").read_text())  # written by live_summ.py from raw reads
live, cached = src["live"], src["cached"]
xmax_data = max([r["step"] for r in live] + [0])
xmax = 1260 if xmax_data <= 1200 else xmax_data * 1.05
cached = [r for r in cached if r["step"] <= max(1200, xmax_data)]


def pts(rows, key):
    return [{"u": r["step"], "y": round(r[key], 3)} for r in rows if r.get(key) is not None]


def end(key):
    if len(live) < 2:
        return None
    last = live[-1]
    lo, hi = last[key + "_ci"]
    return {"u": last["step"], "y": round(last[key], 4), "lo": round(lo, 4), "hi": round(hi, 4)}


series = [
    {"key": "live", "kind": "hero", "label": "fresh thoughts (tonight)", "short": "fresh thoughts",
     "pts": pts(live, "s_P"), "end": end("s_P")},
    {"key": "trace", "kind": "trace", "label": "probe's trace (tonight)", "short": "trace",
     "pts": pts(live, "s_R"), "end": end("s_R")},
    {"key": "cached", "kind": "cached", "label": "fixed thoughts (last night)", "short": "fixed thoughts",
     "pts": pts(cached, "s_P")},
]
# exploratory cell A (DEVIATIONS D4-D5): fixed thoughts plus tonight's thought KL, same pool and seed as last night
kl_p = HERE / "data/runs/cachedkl_s0/reads.jsonl"
if kl_p.exists():
    kl = [json.loads(l) for l in kl_p.read_text().splitlines() if l.strip()]
    if kl and kl[0]["step"] != 0:
        kl = [next(r for r in cached if r["step"] == 0)] + kl
if kl_p.exists() and kl:
    series.insert(2, {"key": "cachedkl", "kind": "kl", "label": "fixed thoughts + thought penalty (tonight)",
                      "short": "fixed + penalty", "pts": pts(kl, "s_P")})
for s in series:
    if s.get("end") is None:
        s.pop("end", None)

onsets = []
for kind, rows, lab, extra in (("cached", cached, "fixed", {"anchor": "end", "dx": -8, "dy": -8}), ("hero", live, "fresh", {})):
    iv = src["onset"].get("cached" if kind == "cached" else "live")
    if iv:
        T = iv[1]
        row = next((r for r in rows if r["step"] == T), None)
        if row:
            onsets.append({"u": T, "y": round(row["s_P"], 4), "label": f"{lab}: {T:,}", "kind": kind, **extra})

fig = {
    "mock": False,
    "title": "On fixed thoughts the word took off at update 600. On fresh thoughts it stayed flat to 1,200.",
    "subtitle": "Qwen3-0.6B, one training run per line. Only the read-or-solve choice is trained. Switch: how much less "
                "the model cheats when the cue is in its thought, as a share of the most it could (100% = it never "
                "cheats then).",
    "x_label": "Training updates →",
    "y_label": "↑ Switch, share of its limit",
    "legend": {
        "band": "take-off window for a match: within 2x of the fixed-thought onset (fixed before the run)",
        "bar": "30%: the bar for learned",
        "end": "update-1,200 test, 95% interval",
    },
    "bands": [{"x0": 300, "x1": 1200, "label": "2x window 300–1,200", "ly": 0.5}],
    "bar": {"y": 0.3},
    "seam": [{"u": 500, "label": "restarts of tonight's run"}, {"u": 800, "label": ""}, {"u": 1000, "label": ""}],
    "onset": onsets,
    "xmax": xmax,
    "series": series,
}
out = HERE / "figures" / "main_figure.json"
out.write_text(json.dumps(fig, indent=1, ensure_ascii=False))
print("wrote", out, "live reads:", len(live), "mock:", fig["mock"])
if BLOG.exists():
    shutil.copy(out, BLOG / "data.json")
    print("copied to", BLOG / "data.json")
