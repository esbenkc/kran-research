"""Gap over training (foldout figure + the 'cached dev gap >= 8' audit trigger).

Cached-input dGap = mean P(X | serving cache) - mean P(X | grading cache) through a checkpoint's merged tail, 200 test
prompts, minus the untrained tail's cached gap (data/base_cached_test.json). Also the DEV curve (64 dev prompts,
every 256 updates, rl_adapt.py dev_curve) minus its step-0 gap: PREREG audits every P checkpoint with cached dev gap
>= 8 points. Sources: data/runs/<run>[_at<step>].json (this project), the U twins' parent stack-conditional jsons
(ref/stack_conditional_runs/), the cleans' parent adaptive-build-switch jsons (ref/parent_data/runs/).
    PY gap_curves.py   -> data/gap_curves.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
b = json.load(open(HERE / "data/base_cached_test.json"))["gap"]


def dg(p):
    t = json.load(open(p))["test_px_cached"]
    return float(np.mean(t["compiled"]) - np.mean(t["eager"])) - b


RUNS = {  # run_id: (kind, dir, file stem, final step)
    "P_s0": ("push", HERE / "data/runs", "P_s0", 16384), "P_s1": ("push", HERE / "data/runs", "P_s1", 16384),
    "R_s0": ("random", HERE / "data/runs", "R_s0", 8192),
    "stack_s0": ("none", HERE / "ref/stack_conditional_runs", "stack_s0_lr0.0001_kl1", 8192),
    "stack_s1": ("none", HERE / "ref/stack_conditional_runs", "stack_s1_lr0.0001_kl1", 8192),
    "cln_s0": ("clean", HERE / "ref/parent_data/runs", "cln_s0", 8192),
    "cln_s1": ("clean", HERE / "ref/parent_data/runs", "cln_s1", 8192),
    "cln_s2": ("clean", HERE / "ref/parent_data/runs", "cln_s2", 8192),
}


def main():
    out = {"note": f"cached-input dGap (fraction), 200 test prompts, minus untrained cached gap {b:+.5f}; "
                   "dev = 64 dev prompts every 256 updates minus the run's step-0 dev gap", "runs": {}}
    for k, (kind, d, stem, final) in RUNS.items():
        pts, last = [], None
        for s in range(1024, final + 1, 1024):
            p = d / (f"{stem}.json" if s == final else f"{stem}_at{s}.json")
            if p.exists():
                pts.append([s, round(dg(p), 5)])
                last = p
        if not pts:
            continue
        row = {"kind": kind, "points": pts, "source": str(d.relative_to(HERE))}
        if kind in ("push", "random") and last is not None:
            dc = json.load(open(last))["dev_curve"]
            g0 = dc[0]["gap"]
            row["dev"] = [[r["step"], round(r["gap"] - g0, 5)] for r in dc if r["step"] > 0]
            row["dev_ge_8_ckpts"] = [s for s, g in row["dev"] if s % 1024 == 0 and g >= 0.08]
        out["runs"][k] = row
        print(k, kind, pts[-3:], "dev>=8 at", row.get("dev_ge_8_ckpts"))
    (HERE / "data/gap_curves.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
