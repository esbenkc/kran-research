"""Builds data.json for the post's main figure (blog.kran.ai assets/figures/llc-early-alarm-8b) from the saved runs only.

Left panel (known-answer test, exploratory walk-length sweep): data/gate/budget.json, mean / min / max over the chain-seed
repeats of each (c, gamma, steps per walk, rank), as a share of Aoyagi's value (236 at rank 1, 312 at rank 8). The
frozen gate grid at 100 steps (data/gate/grid.json, repeat 0, every finished cell) is the strip at 100 steps.
Right panel (Qwen3-8B, exploratory E2): data/llc/e2/*.json, the distance of each walk from its start after 25, 50, 75
and 100 steps, next to the distance a walk of pure noise travels, sqrt(eps * t * d), d = 2,424,832 trained weights.

    python3 figures/build_main.py --out ../../../blog.kran.ai/assets/figures/llc-early-alarm-8b/data.json [--check]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TRUE = {1: 236.0, 8: 312.0}
D_ADAPTER = 2_424_832
SETTINGS = [(0.1, 1.0), (0.1, 0.1), (0.03, 0.1), (0.03, 1.0)]


def left_panel():
    b = json.loads((HERE / "data/gate/budget.json").read_text())
    g = collections.defaultdict(list)
    for x in b["rows"]:
        g[(x["c"], x["gamma"], x["steps"], x["r"])].append(x["lambda_hat"] / TRUE[x["r"]])
    lines = []
    for c, gm in SETTINGS:
        for r in (1, 8):
            for st in b["steps"]:
                v = g[(c, gm, st, r)]
                assert len(v) >= 3, (c, gm, st, r, len(v))
                lines.append({"c": c, "gamma": gm, "rank": r, "steps": st, "y": sum(v) / len(v), "lo": min(v),
                              "hi": max(v), "n": len(v), "truth": TRUE[r]})
    grid = json.loads((HERE / "data/gate/grid.json").read_text())["rows"]
    strip = [{"c": x["c"], "gamma": x["gamma"], "rank": x["r"], "y": x["lambda_hat"] / TRUE[x["r"]],
              "est": x["lambda_hat"]} for x in grid if x["rep"] == 0 and math.isfinite(x["lambda_hat"])]
    n_div = sum(1 for x in grid if x["rep"] == 0 and not math.isfinite(x["lambda_hat"]))
    return lines, strip, n_div


def right_panel():
    walks, cells = [], []
    for f in sorted(glob.glob(str(HERE / "data/llc/e2/m8_*_c*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("burn") != 25 or d.get("draws") != 75:
            continue  # only the 100-step walks of E2 (E2b is a separate, longer run)
        ts = [25, 50, 75, 100][: len(d["dist_from_wstar"][0])]
        for k, dist in enumerate(d["dist_from_wstar"]):
            for t, v in zip(ts, dist):
                noise = math.sqrt(d["eps"] * t * D_ADAPTER)
                walks.append({"c": d["c"], "update": d["step"], "chain": k, "t": t, "dist": v, "noise": noise,
                              "ratio": v / noise})
        cells.append({"c": d["c"], "update": d["step"], "reading": d["lambda_hat"], "se": d["bm_se"],
                      "eps": d["eps"]})
    curves = []
    for c in sorted({w["c"] for w in walks}):
        eps = next(x["eps"] for x in cells if x["c"] == c)
        curves += [{"c": c, "t": t, "noise": math.sqrt(eps * t * D_ADAPTER)} for t in range(0, 101, 5)]
    return walks, cells, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    lines, strip, n_div = left_panel()
    walks, cells, curves = right_panel()
    ratios = [w["ratio"] for w in walks]
    data = {
        "title": "Short walks miss a known answer, and on Qwen3-8B they travel like pure noise",
        "sub_left": "Known-answer test (1,024 weights): at 100 steps no setting lands in the band at both ranks; longer walks at the middle step do",
        "sub_right": "Qwen3-8B adapter (2.4 million weights): each 100-step walk travels exactly as far as pure noise, so the loss did not steer it",
        "n_diverged": n_div,
        "ratio_range": [min(ratios), max(ratios)] if ratios else None,
        "cells": cells,
        "lines": lines, "strip": strip, "walks": walks, "curves": curves,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1))
    print(f"wrote {a.out}: {len(lines)} line points, {len(strip)} strip dots ({n_div} diverged), "
          f"{len(walks)} walk points from {len(cells)} estimates, distance ratio "
          f"{data['ratio_range'][0]:.4f}-{data['ratio_range'][1]:.4f}" if ratios else "no E2 walks")
    if a.check:
        m = {(x["c"], x["gamma"], x["rank"], x["steps"]): x["y"] for x in lines}
        assert abs(m[(0.1, 1.0, 1, 400)] * 236 - 220.8) < 0.1, m[(0.1, 1.0, 1, 400)] * 236
        assert abs(m[(0.1, 1.0, 1, 1600)] * 236 - 246.2) < 0.1
        assert len(strip) == 35 and n_div == 5, (len(strip), n_div)
        best = [x for x in strip if x["c"] == 0.3 and x["gamma"] == 10.0]
        assert sorted(round(x["est"], 1) for x in best) == [196.9, 276.5], best
        print("check OK: 400/1,600-step means 220.8/246.2, 35 finished + 5 diverged gate cells, best setting 196.9/276.5")


if __name__ == "__main__":
    main()
