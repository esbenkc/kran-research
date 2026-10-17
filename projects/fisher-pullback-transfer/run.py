"""Rebuild the post's one figure from the saved results (data/results/pairs.csv + results.json).

Full pipeline (each step writes into data/, see README.md):
    uv run python projects/fisher-pullback-transfer/fetch_data.py
    uv run python projects/fisher-pullback-transfer/build_probes.py
    uv run python projects/fisher-pullback-transfer/fisher.py
    uv run python projects/fisher-pullback-transfer/fisher_big.py
    uv run python projects/fisher-pullback-transfer/select_targets.py
    uv run python projects/fisher-pullback-transfer/build_secondary.py
    (cd projects/fisher-pullback-transfer && uv run python attack.py pilot && ./run_attacks.sh A & ./run_attacks.sh B)
    uv run python projects/fisher-pullback-transfer/kernel_dir.py
    uv run python projects/fisher-pullback-transfer/analysis.py
    uv run python projects/fisher-pullback-transfer/run.py          # this file: figure only
"""
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR.parent.parent))
from lib import figure  # noqa: E402

FIG = "fisher-pullback-transfer"
LABELS = {"plain": "Plain cosine", "whitened": "Whitened", "fisher": "Fisher pull-back"}


def main(publish=False):
    res = json.loads((PROJECT_DIR / "data" / "results" / "results.json").read_text())
    pairs = list(csv.DictReader((PROJECT_DIR / "data" / "results" / "pairs.csv").open()))
    rows = []
    for p in pairs:
        for m in LABELS:
            rows.append({"set": p["set"], "metric": m, "target": p["target"], "probe": p["probe"],
                         "x": round(float(p[m]), 5), "y": round(float(p["measured"]), 5), "n_seeds": int(p["n_seeds"])})
    prim, sec = res["primary"], res.get("secondary", {})
    data = {
        "title": "Which other probes does a hiding fine-tune fool? The formula's forecast vs what happened",
        "sub": ("Each dot is one probe after a fine-tune of Pythia-70M hid one concept from one target probe "
                "while keeping its outputs close to the original. Across: AUROC drop predicted by one formula, "
                "with three choices of geometry. Up: measured AUROC drop. Dashed line: prediction = measurement. "
                "R² = 1 − (squared error)/(variance of the measurements); 1 is perfect, below 0 is worse than "
                "guessing the average. Filled and hollow sets are scored separately, never pooled. Each dot is the mean "
                "of the target's successful seeds (2-3). Both axes stretch near zero (symmetric log scale)."),
        "scale": "symlog", "symlog_constant": 0.01,
        "ticks": [-0.1, 0, 0.1, 0.5],
        "panels": [{"metric": m, "label": LABELS[m], "r2_primary": prim[m], "r2_secondary": sec.get(m)}
                   for m in LABELS],
        "stats": {"n_primary": prim["n_pairs"], "sd_primary": prim["sd_measured"],
                  "n02_primary": prim["n_abs_drop_gt_0.02"], "n_secondary": sec.get("n_pairs", 0),
                  "sd_secondary": sec.get("sd_measured"), "n02_secondary": sec.get("n_abs_drop_gt_0.02")},
        "rows": rows,
    }
    fig = figure.save(PROJECT_DIR, FIG, data, (PROJECT_DIR / "plot.js").read_text())
    print("wrote", fig)
    if publish:  # into the work clone next to kran-research, never the Desktop blog
        figure.publish(fig, PROJECT_DIR.parent.parent.parent / "blog.kran.ai" / "assets" / "figures")


if __name__ == "__main__":
    main(publish="--publish" in sys.argv)
