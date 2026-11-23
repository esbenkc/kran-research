"""results/analysis.json -> the main figure's data (blog.kran.ai/assets/figures/anchor-text-law/data.json).

    uv run python figures/figure_data.py [--blog ../../../blog.kran.ai]

Writes ONLY the number keys (rows, g_fired, fr_gap, anchor_gap, status, models, n); text keys that already exist in the
file (title, subtitle, answer, note, groups, row labels) are Selma's and are kept as they are.
verdict per held-out row: 'held' if |err| <= 1/10 (point), 'miss' if the 95% interval of err lies wholly outside
[-1/10, +1/10] (strict), else 'off'. Fit rows: 'fit'.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as F
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
NAMES = {"gpt-oss-20b": "gpt-oss-20b", "qwen3-8b-nothink": "Qwen3-8B", "qwen3-8b": "Qwen3-8B"}
HELDOUT = {("A", "md"), ("A", "anchor"), ("B", "raw"), ("B", "bare")}


def frac(x):
    return None if x is None else F(x["frac"])


def val(x):
    return None if x is None else x["value"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(PROJ.parent.parent.parent / "blog.kran.ai"))
    ap.add_argument("--analysis", default=str(PROJ / "results" / "analysis.json"))
    a = ap.parse_args()
    res = json.loads(Path(a.analysis).read_text())
    models = {m: NAMES.get(v, v) for m, v in res["models"].items()}
    rows = []
    for c in res["cells"]:
        if not c["scored"]:
            continue
        m, f = c["slot"], c["cell"].split("-")[0]
        row = {"model": models[m], "format": f, "group": "heldout" if (m, f) in HELDOUT else "fit",
               "n": c["n"], "followed": c["followed"], "measured": val(c["catch"]),
               "lo": val(c["catch_ci"][0]), "hi": val(c["catch_ci"][1]),
               "shown": val(c["reach"]), "reached": c["reached"], "fgr": val(c["fr"]),
               "fgr_lo": val(c["fr_ci"][0]), "fgr_hi": val(c["fr_ci"][1]),
               "status_cell": (m, f) in {("A", "md"), ("A", "anchor"), ("B", "raw")}}
        if row["group"] == "heldout":
            h = res["heldout"][f"{m}-{f}"]
            err, ci = frac(h["err"]), [frac(x) for x in h["err_ci"]]
            row.update(pred=val(h["pred"]), reach=val(h["reach_from"]), follow=val(h["follow"]), err=val(h["err"]),
                       err_ci=[val(x) for x in h["err_ci"]],
                       err_seed_ci=list(h["err_seed"]["t"][1:]) if h.get("err_seed") and h["err_seed"]["t"] else None,
                       verdict="held" if abs(err) <= F(1, 10) else ("miss" if ci[1] < -F(1, 10) or ci[0] > F(1, 10) else "off"))
        else:
            row["verdict"] = "fit"
        rows.append(row)
    st = res["status"]
    out_path = Path(a.blog) / "assets" / "figures" / "anchor-text-law" / "data.json"
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    data.update({
        "rows": rows, "status": st["status"], "models": models, "n": res["floor"],
        "g_fired": {models[m]: v for m, v in st.get("G_fired", {}).items()},
        "fr_gap": {models[m]: val(d["bare_md"]["value"]) for m, d in res["fr_diffs"].items()},
        "anchor_gap": {models[m]: val(d["anchor_md"]["value"]) for m, d in res["fr_diffs"].items()},
        "anchor_gap_ci": {models[m]: [val(x) for x in d["anchor_md"]["ci"]] for m, d in res["fr_diffs"].items()},
        "fr_gap_ci": {models[m]: [val(x) for x in d["bare_md"]["ci"]] for m, d in res["fr_diffs"].items()},
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", out_path, len(rows), "rows; status", st["status"])


if __name__ == "__main__":
    main()
