"""data.json for the main figure (blog.kran.ai/assets/figures/lie-erasure-cost/plot.js) from data/results/figure_data.json
and data/results/still_lying.json. Panel titles carry each concept's LEACE price (measured before training).
Usage: venv/bin/python blog_data.py ../../../blog.kran.ai/assets/figures/lie-erasure-cost/data.json
"""
import json
import sys
from pathlib import Path

D = Path(__file__).parent / "data"


def main(out):
    f = json.loads((D / "results" / "figure_data.json").read_text())
    pr = f["prices"]
    sl = D / "results" / "still_lying.json"
    still = json.loads(sl.read_text()) if sl.exists() else None
    title = {"lie": f"The lie (LEACE cost {pr['lie_blind']:.2f})",
             "decoy": "The decoy word (LEACE cost at noise level)"}
    runs = []
    for r in sorted(f["runs"], key=lambda r: r["target"] != "lie"):
        row = {"key": r["target"], "panel": title[r["target"]], "steps": r["steps"], "blind": r["catch_blind"],
               "full": r["catch_full"], "out": r["O_catch"], "late": [r["late_steps"][0], r["late_steps"][-1]],
               "L": r["L_blind"], "gap": len(r["steps"]) > 1 and r["steps"][1] - r["steps"][0] > 50,
               # the pre-registered bar of each run (v1.3 s7): lie held = L >= 50%; decoy fell = L <= step-0 catch - 30 pts
               "bar": ({"y": 0.5, "label": "our 50% bar"} if r["target"] == "lie" else
                       {"y": r["catch_blind"][0] - 0.30, "label": f"our {round((r['catch_blind'][0] - 0.30) * 100)}% bar"})}
        vals = [x for k in ("catch_blind", "catch_full", "O_catch") for x in r[k] if x is not None]
        if r["target"] == "lie" and min(vals) >= 0.99:
            row["note"] = f"{round(r['L_blind'] * 100)}% caught, late mean\n(all three probes: 99-100% at every step)"
        if r["target"] == "lie" and still:
            st = sorted(still["steps"].items(), key=lambda kv: int(kv[0]))
            row["still"] = {"steps": [int(k) for k, _ in st], "v": [v["still_false"] for _, v in st]}
        runs.append(row)
    data = {"title": "Probes retrained during each fine-tune: share of held-out prompts caught",
            "subtitle": "Base Qwen2.5-1.5B, 800 prompts from 200 held-out cities. Alarm line: 1% of prompts without the concept cross it.",
            "runs": runs,
            "legend": {"blind": "output-blind probes (country-score directions removed)", "full": "probes on the whole activation",
                       "out": "output probe (country scores)", "still": "lie prompts: model still names a wrong country",
                       "still_short": "lie prompts where the model still lies",
                       "gap": "not scored at steps 50 and 100"},
            "note": None}
    Path(out).write_text(json.dumps(data, indent=1))
    print("wrote", out, [(r["key"], r["L"]) for r in runs])


if __name__ == "__main__":
    main(sys.argv[1])
