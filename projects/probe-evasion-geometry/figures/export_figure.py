"""Export the ONE figure's data.json from the frozen decision runs (Mahalanobis retain, 3 seeds).

Run from kran-research/:  uv run python projects/probe-evasion-geometry/figures/export_figure.py
Writes projects/probe-evasion-geometry/figures/probe-evasion-cosine/data.json and copies it to the
blog clone (../blog.kran.ai/assets/figures/probe-evasion-cosine/data.json) when that folder exists.
"""

import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "runs"
OUT = HERE / "probe-evasion-cosine" / "data.json"
BLOG = HERE.parents[3] / "blog.kran.ai" / "assets" / "figures" / "probe-evasion-cosine"
SETTINGS = [("synthetic", "Synthetic Gaussian data (64 dims)"), ("lm", "Pythia-70M activations (layer 3)")]


def r2(x, y):
    return float(np.corrcoef(x, y)[0, 1] ** 2)


panels, rows = [], []
for key, label in SETTINGS:
    summ = [s for s in json.loads((RUNS / f"summary_{key}.json").read_text()) if s["retain"] == "mahalanobis"
            and not s.get("early_stop")]
    assert len(summ) == 3, (key, len(summ))
    rw, rr = [], []
    for s in summ:
        probes = json.loads((RUNS / f"{key}_mahalanobis_seed{s['seed']}_probes.json").read_text())
        assert len(probes) == 50
        x_w = np.array([p["pred_drop_white"] for p in probes])
        x_r = np.array([p["pred_drop_raw"] for p in probes])
        y = np.array([p["drop"] for p in probes])
        rw.append(r2(x_w, y))
        rr.append(r2(x_r, y))
        assert abs(rw[-1] - s["r2_white"]) < 1e-6 and abs(rr[-1] - s["r2_raw"]) < 1e-6
        for p in probes:
            for series, xk in (("lemma", "pred_drop_white"), ("raw", "pred_drop_raw")):
                rows.append({
                    "setting": key, "seed": s["seed"], "probe": p["probe"], "kind": p["kind"],
                    "dim": p["subspace_dim"], "series": series, "x": round(p[xk], 5),
                    "y": round(p["drop"], 5), "before": round(p["auroc_before"], 4),
                    "after": round(p["auroc_after"], 4), "cos_white": round(p["cos_white"], 3),
                    "cos_raw": round(p["cos_raw"], 3),
                })
    panels.append({
        "setting": key, "label": label,
        "r2_white": round(float(np.mean(rw)), 3), "r2_raw": round(float(np.mean(rr)), 3),
        "r2_white_seeds": [round(v, 3) for v in rw], "r2_raw_seeds": [round(v, 3) for v in rr],
        "refit": round(float(np.mean([s["auroc_refit"] for s in summ])), 3),
        "w_before": round(float(np.mean([s["auroc_w_before"] for s in summ])), 3),
        "w_after": round(float(np.mean([s["auroc_w_after"] for s in summ])), 3),
    })

data = {
    "title": "One angle predicts which probes the evader also fools",
    "sub": "Each dot is one of 50 probes in one of 3 seeds. Across: drop in AUROC that the formula predicts. "
           "Up: drop we measured after the evader trained against one probe. On the dashed line, prediction = measurement. r² = share of the variance in measured drops that the prediction explains, mean of 3 seeds; refit = AUROC of a fresh probe trained after the evasion.",
    "panels": panels,
    "rows": rows,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1))
print(json.dumps(panels, indent=1))
if BLOG.parent.exists():
    BLOG.mkdir(exist_ok=True)
    shutil.copy(OUT, BLOG / "data.json")
    print("copied to", BLOG)
