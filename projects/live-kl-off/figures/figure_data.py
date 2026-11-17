"""Build data.json for the main figure of live-kl-off (Selma).

One panel: switch (s_P, share of its limit) vs training updates for
  - tonight's fresh-thought seeds WITHOUT the thought penalty (data/runs/kl0_s*/reads.jsonl), one line per seed;
  - the earlier fresh-thought run WITH the penalty (parent on-policy-planted, live_k4);
  - the earlier fixed-thought run (parent: decision-only-control d010_s0, thoughts sampled once and reused);
  - the earlier fixed-thought run with the penalty (parent cell A, cachedkl_s0, exploratory).
Parent rows come from figures/parent_data.json, a copy of the parent's figures/data.json written by
live_summ.py from its raw reads (provenance in that file's "_source").
Onsets: from analysis.json (the frozen rule with its paired interval) when present, else the two-reads >= 0.30
part of the rule only, marked provisional.

    /usr/bin/python3 figures/figure_data.py
Writes figures/main_figure.json and copies it to the blog clone's assets/figures/live-kl-off/data.json.
"""
import glob
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "live-kl-off"
PARENT_REF = "origin/swarm/oracle-1113-on-policy-planted"
PARENT_PATH = "projects/on-policy-planted/figures/data.json"
BAR = 0.30

pp = HERE / "figures" / "parent_data.json"
if not pp.exists():
    sha = subprocess.check_output(["git", "rev-parse", PARENT_REF], cwd=HERE, text=True).strip()
    raw = json.loads(subprocess.check_output(["git", "show", f"{sha}:{PARENT_PATH}"], cwd=HERE, text=True))
    raw["_source"] = f"esbenkc/kran-research {sha[:9]}:{PARENT_PATH} (live_summ.py from the parent's raw reads)"
    pp.write_text(json.dumps(raw, indent=1))
parent = json.loads(pp.read_text())


def rows(path):
    out = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return sorted({r["step"]: r for r in out}.values(), key=lambda r: r["step"])


def pts(rs, key="s_P"):
    return [{"u": r["step"], "y": round(r[key], 4)} for r in rs if r.get(key) is not None]


def onset_simple(rs):
    """First read with s_P >= BAR at it and at the next read (no interval check)."""
    for a, b in zip(rs, rs[1:]):
        if a["s_P"] >= BAR and b["s_P"] >= BAR:
            return a["step"]
    return None


an_p = HERE / "analysis.json"
an = json.loads(an_p.read_text()) if an_p.exists() else {}
seeds = []
for f in sorted(glob.glob(str(HERE / "data/runs/kl0_s*/reads.jsonl"))):
    m = re.search(r"kl0_s(\d+)", f)
    rs = rows(f)
    if rs:
        seeds.append((int(m.group(1)), rs))

series, onsets = [], []
for i, (sd, rs) in enumerate(seeds):
    series.append({"key": f"kl0_s{sd}", "kind": "hero", "seed": sd,
                   "label": f"fresh, no penalty, seed {sd + 1} (tonight)" if i == 0 else f"seed {sd + 1}",
                   "short": f"seed {sd + 1}", "pts": pts(rs)})
    a_on = (an.get("seeds", {}).get(f"kl0_s{sd}", {}) or {}).get("onset")  # [lo, hi] interval from the frozen rule
    T = a_on[1] if a_on else None
    prov = False
    if T is None and not an:
        T = onset_simple(rs)
        prov = T is not None
    if T is not None:
        row = next((r for r in rs if r["step"] == T), None)
        if row:
            onsets.append({"u": T, "y": round(row["s_P"], 4), "label": f"{T:,}" + ("?" if prov else ""), "kind": "hero"})

live = parent["live"]
cached = [r for r in parent["cached"] if r["step"] <= 1200]
ckl = parent.get("diagnostics", {}).get("cachedkl_s0", [])
series += [
    {"key": "live_kl", "kind": "klon", "label": "fresh + penalty (earlier)", "short": "fresh + penalty", "pts": pts(live)},
    {"key": "cached", "kind": "cached", "label": "fixed thoughts (earlier)", "short": "fixed", "pts": pts(cached)},
]
if ckl:
    series.append({"key": "cached_kl", "kind": "ckl", "label": "fixed thoughts + penalty (earlier)",
                   "short": "fixed + penalty", "pts": pts(ckl)})
c_on = parent.get("onset", {}).get("cached")
if c_on:
    row = next((r for r in cached if r["step"] == c_on[1]), None)
    if row:
        onsets.append({"u": c_on[1], "y": round(row["s_P"], 4), "label": f"{c_on[1]:,}", "kind": "cached",
                       "anchor": "end", "dx": -8, "dy": -8})

fig = {
    "mock": not seeds,
    "title": "[FILL result title, set from the numbers]",
    "subtitle": "Qwen3-0.6B. Only the read-or-solve choice is trained. Switch: how much less the model cheats when "
                "the word is in its thought, as a share of the most it could (100% = it never cheats then).",
    "x_label": "Training updates →",
    "y_label": "↑ Switch, share of its limit",
    "legend": {
        "band": "predicted take-off if the penalty caused the whole delay (fixed before the run)",
        "bar": "30%: the bar for a take-off",
    },
    "bands": [{"x0": 500, "x1": 700, "label": "500–700", "ly": 0.97}],
    "bar": {"y": BAR},
    "onset": onsets,
    "xmax": 1260,
    "series": series,
}
out = HERE / "figures" / "main_figure.json"
out.write_text(json.dumps(fig, indent=1, ensure_ascii=False))
print("wrote", out, "seeds:", [s for s, _ in seeds], "last reads:", [r[-1]["step"] for _, r in seeds],
      "onsets:", [(o["kind"], o["u"]) for o in onsets], "mock:", fig["mock"])
if BLOG.exists():
    shutil.copy(out, BLOG / "data.json")
    print("copied to", BLOG / "data.json")
