"""Main-figure data: results/summary.json -> blog assets/figures/discoverable-route/data.json (numbers only).

python figure_data.py [--smoke] [--out PATH]
Text fields (title, subtitle, note, label, sub) are Selma's and are kept from the existing data.json; this script
writes every number. named and runbook use all their seeds (0..39, the status block); linked and none use 0..29.
p = equal-weight mean over the five models; lo/hi = the seed-joint bootstrap 95% CI from analysis.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BLOG = PROJECT.parents[2] / "blog.kran.ai" / "assets" / "figures" / "discoverable-route" / "data.json"
ORDER = ["claude-haiku-4-5-20251001", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gpt-5.4-mini", "gpt-5.4-nano"]
CELL_ORDER = ["named", "runbook", "linked", "none"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=BLOG)
    a = ap.parse_args()
    s = json.loads((PROJECT / ("results_smoke" if a.smoke else "results") / "summary.json").read_text())
    old = json.loads(a.out.read_text()) if a.out.exists() else {"cells": []}
    text = {c["id"]: c for c in old.get("cells", [])}
    pt, ci = s["main"]["point"], s["main"]["ci95"]
    pm = s["per_model_cells"]
    src = {"named": ("p_named_key", "p_named_key"), "runbook": ("p_runbook_key", "p_runbook_key"),
           "linked": ("p_linked", "p_linked"), "none": ("p_none", "p_none")}
    cells = []
    for c in CELL_ORDER:
        models = []
        for m in ORDER:
            d = pm[f"{m}|{c}"]
            if c in ("linked", "none"):
                assert d["seeds"][1] < 30 or a.smoke, (c, m, d["seeds"])
            models.append({"model": s["names"][m], "k": d["key"], "n": d["n"], "p": round(d["key"] / d["n"], 4)})
        k, n = sum(x["k"] for x in models), sum(x["n"] for x in models)
        p = sum(pm[f"{m}|{c}"]["key"] / pm[f"{m}|{c}"]["n"] for m in ORDER) / len(ORDER)
        key_pt, key_ci = src[c]
        assert abs(p - pt[key_pt]) < 1e-9, (c, p, pt[key_pt])  # the figure's dot = analysis.py's pooled rate
        t = text.get(c, {})
        cells.append({"id": c, "label": t.get("label", c), "sub": t.get("sub", ""), "k": k, "n": n,
                      "p": round(p, 4), "lo": round(ci[key_ci][0], 4), "hi": round(ci[key_ci][1], 4),
                      "pred": round(pt["law_pred_linked"], 4) if c == "linked" else None,
                      "band_from": "named" if c == "runbook" else None, "models": models})
    out = {"title": old.get("title", ""), "subtitle": old.get("subtitle", ""), "note": old.get("note", ""),
           "edge": s["edge"], "cells": cells,
           "source": {"summary": "kran-research projects/discoverable-route/results/summary.json",
                      "status": s["status"], "delta": round(pt["delta"], 4),
                      "delta_ci95": [round(x, 4) for x in ci["delta"]],
                      "law_d_linked": round(pt["law_d_linked"], 4), "p_open_linked": round(pt["p_open_linked"], 4)}}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}: " + ", ".join(f"{c['id']} {c['k']}/{c['n']} p={c['p']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]" for c in cells))


if __name__ == "__main__":
    main()
