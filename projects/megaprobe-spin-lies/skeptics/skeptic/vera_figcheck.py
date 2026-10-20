"""Check every point of the figure's data.json against my own AUROCs from build's saved per-row scores.
Arm map: a -> a_<fit>, b -> b2_<fit> (D7 refit), c -> c_<fit>, c2 -> c2_<fit>. trained = mean of the 4 trained models.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_figcheck.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import TRAINED, auroc, classes  # noqa: E402
from vera_rebuild import load_fit  # noqa: E402  (my loader; running it also re-prints the rebuild)

FIG = Path("/Users/esben/work/oracle-1020-megaprobe-spin-lies/blog.kran.ai/assets/figures/megaprobe-spin-lies/data.json")
fig = json.load(open(FIG))
AM = {"a": "a", "b": "b2", "c": "c", "c2": "c2"}
F = {m: load_fit(m) for m in ["base"] + TRAINED}


def val(model, arm, fit):
    S, L = F[model]; c = classes(L); k = f"{AM[arm]}_{fit}"
    if k not in S:
        return None
    return auroc(S[k][c["spin"]], S[k][c["truthful_fs"]])


worst = 0
print("\n--- figure check ---")
for p in fig["points"]:
    if p["series"] == "base":
        mine = val("base", p["arm"], p["fit"])
    else:
        vs = [val(m, p["arm"], p["fit"]) for m in TRAINED]
        mine = None if None in vs else float(np.mean(vs))
    d = None if mine is None else abs(mine - p["auroc"])
    worst = max(worst, d or 0)
    print(p["series"], p["arm"], p["fit"], "fig %.4f mine %s diff %s" % (p["auroc"], "NA" if mine is None else "%.4f" % mine, "NA" if d is None else "%.1e" % d))
for p in fig.get("per_model", []):
    mine = val(p["model"], p["arm"], p.get("fit", "all"))
    d = abs(mine - p["auroc"]); worst = max(worst, d)
    print("per_model", p["model"], p["arm"], "fig %.4f mine %.4f diff %.1e" % (p["auroc"], mine, d))
print("WORST |diff| = %.2e" % worst)
