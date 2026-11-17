"""Build data.json for the main figure of live-kl-off (Selma).

One panel: switch (s_P, share of its limit) vs training updates for
  - tonight's fresh-thought seeds WITHOUT the thought penalty, one line per seed;
  - the earlier fresh-thought run WITH the penalty (parent on-policy-planted, live_k4);
  - the earlier fixed-thought run (decision-only-control d010_s0, thoughts sampled once and reused);
  - the earlier fixed-thought run with the penalty (parent cell A, cachedkl_s0, exploratory, cut at 876);
  - the dense-word positive control (cue 0.86, no penalty), when it ran.
Every number comes from data/summary.json, which kl_summ.py writes from the raw reads (parent reads are
byte-identical copies under data/parent/). Onsets are kl_summ.py's frozen-rule intervals.

    /usr/bin/python3 kl_summ.py && /usr/bin/python3 figures/figure_data.py
Writes figures/main_figure.json and copies it to the blog clone's assets/figures/live-kl-off/data.json.
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "live-kl-off"
BAR = 0.30

sm_p = HERE / "data" / "summary.json"  # kl_summ.py, from the raw reads (kl-off seeds, parent lines, positive control)
sm = json.loads(sm_p.read_text())
an_p = HERE / "analysis.json"
an = json.loads(an_p.read_text()) if an_p.exists() else {}


def pts(rs, key="s_P"):
    return [{"u": r["step"], "y": round(r[key], 4)} for r in rs if r.get(key) is not None]


def onset_dot(rs, on, kind, label=None, **extra):
    if not on:
        return None
    row = next((r for r in rs if r["step"] == on[1]), None)
    if not row:
        return None
    return {"u": on[1], "y": round(row["s_P"], 4), "label": label or f"{on[1]:,}", "kind": kind, **extra}


series, onsets = [], []
kl = sorted(sm["seeds"], key=lambda r: r["name"])
for i, r in enumerate(kl):
    sd = int(re.search(r"kl0_s(\d+)", r["name"]).group(1))
    series.append({"key": r["name"], "kind": "hero", "seed": sd,
                   "label": f"fresh, no penalty, seed {sd} (tonight)" if i == 0 else f"seed {sd}",
                   "short": f"no penalty, seed {sd}" if i == 0 else f"seed {sd}", "pts": pts(r["reads"]),
                   "end": {"u": r["last_read"], "y": round(r["last"]["s_P"]["v"], 4), "lo": round(r["last"]["s_P"]["ci"][0], 4),
                           "hi": round(r["last"]["s_P"]["ci"][1], 4)}})
    onsets.append(onset_dot(r["reads"], r["onset"], "hero"))
par = sm["parent"]
if "live_k4" in par:
    series.append({"key": "live_kl", "kind": "klon", "label": "fresh + penalty (earlier)", "short": "fresh + penalty",
                   "pts": pts(par["live_k4"]["reads"])})
if "cached_d010_s0" in par:
    rs = [x for x in par["cached_d010_s0"]["reads"] if x["step"] <= 1200]
    series.append({"key": "cached", "kind": "cached", "label": "fixed thoughts (earlier)", "short": "fixed", "pts": pts(rs)})
    onsets.append(onset_dot(rs, par["cached_d010_s0"]["onset"], "cached", anchor="end", dx=-8, dy=-8))
if "cachedkl_s0" in par:
    rs = par["cachedkl_s0"]["reads"]
    if rs and rs[0]["step"] != 0 and "cached_d010_s0" in par:
        rs = [par["cached_d010_s0"]["reads"][0]] + rs
    series.append({"key": "cached_kl", "kind": "ckl", "label": "fixed thoughts + penalty (earlier, cut at 876)",
                   "short": "fixed + penalty", "pts": pts(rs)})
pc = sm.get("positive_control")
if pc:
    series.append({"key": "posctl", "kind": "pos", "label": "dense word, 86 in 100, no penalty (control)",
                   "short": "dense control", "pts": pts(pc["reads"])})
onsets = [o for o in onsets if o]
seeds = kl

fig = {
    "mock": not seeds,
    "title": "Without the penalty, seed 0 stayed below 2% to update 1,200. Fixed thoughts took off at 600.",
    "subtitle": "Qwen3-0.6B. Only the read-or-solve choice is trained. Switch: how much less the model cheats when "
                "the word is in its thought, as a share of the most it could (100% = it never cheats then).",
    "x_label": "Training updates →",
    "y_label": "↑ Switch, share of its limit",
    "legend": {
        "band": "predicted take-off if the penalty caused the whole delay (fixed before the run)",
        "bar": "30%: the bar for a take-off",
        "end": "last test of tonight's seed, 95% interval",
    },
    "bands": [{"x0": 500, "x1": 700, "label": "500–700", "ly": 0.97}],
    "bar": {"y": BAR},
    "onset": onsets,
    "xmax": 1260,
    "series": series,
}
out = HERE / "figures" / "main_figure.json"
out.write_text(json.dumps(fig, indent=1, ensure_ascii=False))
print("wrote", out, "seeds:", [r["name"] for r in seeds], "last reads:", [r["last_read"] for r in seeds],
      "onsets:", [(o["kind"], o["u"]) for o in onsets], "mock:", fig["mock"])
if BLOG.exists():
    shutil.copy(out, BLOG / "data.json")
    print("copied to", BLOG / "data.json")
