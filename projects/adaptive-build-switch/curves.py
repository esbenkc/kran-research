"""Gap over training for the main figure's left panel (Drew): data/fig_curves.json.

Gap at a checkpoint = mean P(flatter) on the 200 test prompts, serving minus grading, read on the CACHED block-26
inputs (the quick reading every run saves: test_px_cached in data/runs/<run>[_at<step>].json), minus the untrained
model's cached gap (data/base_cached_test.json). The whole-graph dGap (status number) is read only at the end
(summary_models.json); the panel note says the curve is the quick reading.
Twins (not trained to hide) are the parent stack-conditional runs, reused unchanged (PREREG v1 models table);
their checkpoint files are read from PARENT and every source file's sha256 is recorded.
    uv run python curves.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "data/runs"
PARENT = Path.home() / "work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data/runs"
OUT = HERE / "data/fig_curves.json"
RUNS_OWN = [("adv_mid_s0", "hide", "Trained to hide, run 1"), ("adv_mid_s1", "hide", "Trained to hide, run 2"),
            ("adv_mid_s1x", "hide_cont", "Trained to hide, run 2, continued"),
            ("cln_s0", "clean", "Clean 1"), ("cln_s1", "clean", "Clean 2"), ("cln_s2", "clean", "Clean 3")]
RUNS_PARENT = [("stack_s0_lr0.0001_kl1", "plain", "Not trained to hide, run 1"),
               ("stack_s1_lr0.0001_kl1", "plain", "Not trained to hide, run 2")]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def gap_of(p, base_gap, ids_ref):
    d = json.loads(p.read_text())
    t = d["test_px_cached"]
    assert d["test_ids"] == ids_ref, f"{p.name}: test ids differ"
    return float(np.mean(t["compiled"]) - np.mean(t["eager"]) - base_gap)


def points(folder, run, base_gap, ids_ref, final_step=8192):
    pts, src = [], {}
    for p in sorted(folder.glob(f"{run}_at*.json"), key=lambda p: int(p.stem.split("_at")[1])):
        if "adapter" in p.name:
            continue
        s = int(p.stem.split("_at")[1])
        pts.append([s, round(100 * gap_of(p, base_gap, ids_ref), 2)])
        src[p.name] = sha(p)
    fin = folder / f"{run}.json"
    if fin.exists():
        d = json.loads(fin.read_text())
        s = int(d.get("args", {}).get("steps", final_step))
        if not pts or pts[-1][0] < s:
            pts.append([s, round(100 * gap_of(fin, base_gap, ids_ref), 2)])
            src[fin.name] = sha(fin)
    return pts, src


def main():
    b = json.loads((HERE / "data/base_cached_test.json").read_text())
    base_gap = float(b["gap"])
    ids_ref = json.loads((RUNS / "adv_mid_s0.json").read_text())["test_ids"]
    lines, sources = [], {"data/base_cached_test.json": sha(HERE / "data/base_cached_test.json")}
    for run, kind, label in RUNS_OWN:
        pts, src = points(RUNS, run, base_gap, ids_ref)
        if pts:
            lines.append({"run": run, "kind": kind, "label": label, "points": [[0, 0.0]] + pts})
            sources.update({f"data/runs/{k}": v for k, v in src.items()})
    for run, kind, label in RUNS_PARENT:
        pts, src = points(PARENT, run, base_gap, ids_ref)
        lines.append({"run": run.split("_lr")[0], "kind": kind, "label": label, "points": [[0, 0.0]] + pts})
        sources.update({f"parent stack-conditional data/runs/{k}": v for k, v in src.items()})
    pilot = RUNS / "pilot_cap1_s0.json"
    pp = {"run": "pilot_cap1_s0", "kind": "pilot", "label": "weight cap only",
          "points": [[4096, round(100 * gap_of(pilot, base_gap, ids_ref), 2)]]}
    sources["data/runs/pilot_cap1_s0.json"] = sha(pilot)
    lines.append(pp)
    OUT.write_text(json.dumps({"reading": "cached block-26 inputs, 200 test prompts, minus untrained cached gap, points",
                               "base_cached_gap": base_gap, "lines": lines, "sources": sources}, indent=1) + "\n")
    for l in lines:
        print(f"{l['run']:14s} {l['kind']:9s} " + " ".join(f"{s}:{g:+.1f}" for s, g in l["points"]))


if __name__ == "__main__":
    main()
