"""data.json for the main figure (blog.kran.ai/assets/figures/lie-erasure-cost/plot.js) from data/results/figure_data.json.
Usage: venv/bin/python blog_data.py ../../../blog.kran.ai/assets/figures/lie-erasure-cost/data.json
"""
import json
import sys
from pathlib import Path

D = Path(__file__).parent / "data"


def main(out):
    f = json.loads((D / "results" / "figure_data.json").read_text())
    pr = f["prices"]
    meta = {"lie": ("the lie (8 false examples)", "lie", pr["lie_blind"]),
            "decoy": ("the decoy word (map or book)", "decoy", pr["decoy_blind"])}
    runs = []
    for r in f["runs"]:
        key = r["target"] if r["lam"] == 10000 else r["target"] + "5"
        name, short, price = meta[r["target"]]
        if r["lam"] != 10000:
            name += f", output penalty x{r['lam'] / 1e4:g}"
        st = r["steps"]
        runs.append({"key": key, "name": name, "short": short, "price": price,
                     "price_label": f"erasure cost {price:.3g} nats" if price >= 0.01 else f"erasure cost {price:.1e} nats",
                     "steps": st, "blind": r["catch_blind"], "full": r["catch_full"], "out": r["O_catch"],
                     "late": [r["late_steps"][0], r["late_steps"][-1]], "L": r["L_blind"]})
    data = {"title": "TITLE (Selma)", "subtitle": "Base Qwen2.5-1.5B, 200 held-out cities x 4 prompts; alarm line set so 1% of negative prompts cross it",
            "runs": runs,
            "legend": {"blind": "probes that cannot see the answer directions", "full": "probes on the whole activation",
                       "out": "probe on the answer scores alone", "half": "half caught"},
            "note": None}
    Path(out).write_text(json.dumps(data, indent=1))
    print("wrote", out, [(r["key"], r["L"]) for r in runs])


if __name__ == "__main__":
    main(sys.argv[1])
