"""Main-figure data from results/scores.jsonl. python figure_data.py [out.json ...]

Default outs: results/figure_data.json and ../../../blog.kran.ai/assets/figures/warning-override/data.json (same content).
Per cell (warning off/on x override absent/present): model-equal share of runs that used the key, the flag, either
(any out-of-scope), and the route mix (key only / flag only / both / none), each with a 90% bootstrap CI
(runs resampled within model x cell, B = 10,000, default_rng(3)); plus each model's own shares (the dots).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = {"claude-haiku-4-5-20251001": "Claude Haiku 4.5", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
         "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano"}
CELLS = ("off_absent", "on_absent", "off_present", "on_present")
TAGS = ("key", "flag", "any", "key_only", "flag_only", "both", "none")


def tags(r: dict) -> list[int]:
    return [int(r["key"]), int(r["flag"]), int(r["any"]), int(r["key"] and not r["flag"]),
            int(r["flag"] and not r["key"]), int(r["key"] and r["flag"]), int(not r["any"])]


def main() -> None:
    outs = [Path(p) for p in sys.argv[1:]] or [ROOT / "results" / "figure_data.json",
                                                ROOT.parents[2] / "blog.kran.ai" / "assets" / "figures" / "warning-override" / "data.json"]
    rows = [json.loads(ln) for ln in (ROOT / "results" / "scores.jsonl").read_text().splitlines()]
    by: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["seed"]):
        by[(r["model"], r["cell"])].append(tags(r))
    models = sorted({m for m, _ in by})
    rng = np.random.default_rng(3)
    out = {"source": "kran-research projects/warning-override results/scores.jsonl (figure_data.py)", "n_per_cell_per_model": 40,
           "models": [NAMES[m] for m in models], "cells": {}}
    for c in CELLS:
        arrs = [np.array(by[(m, c)]) for m in models]
        point = np.mean([a.mean(axis=0) for a in arrs], axis=0)
        draws = np.mean([a[rng.integers(0, len(a), size=(10_000, len(a)))].mean(axis=1) for a in arrs], axis=0)
        lo, hi = np.percentile(draws, [5, 95], axis=0)
        out["cells"][c] = {
            "warning": c.startswith("on"), "override_offered": c.endswith("present"),
            **{t: {"share": round(float(point[i]), 4), "ci90": [round(float(lo[i]), 4), round(float(hi[i]), 4)]}
               for i, t in enumerate(TAGS)},
            "per_model": {NAMES[m]: {t: round(float(a.mean(axis=0)[i]), 4) for i, t in enumerate(TAGS)} for m, a in zip(models, arrs)},
        }
    text = json.dumps(out, indent=1) + "\n"
    for o in outs:
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(text)
        print("wrote", o)
    for c in CELLS:
        v = out["cells"][c]
        print(c, {t: v[t]["share"] for t in TAGS})


if __name__ == "__main__":
    main()
