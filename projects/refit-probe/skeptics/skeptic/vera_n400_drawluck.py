"""Is build's 20-draw mean a draw-luck outlier? 300 fresh class-balanced draws of mine per cell.
    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_n400_drawluck.py"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, fit_lr, score, balanced_draw
from vera_rebuild import load
rows = [json.loads(l) for l in open(__import__("vera_rebuild").D / "results/draws.jsonl")]
for p, n in [("lam1_s0", 400), ("lam1_s0", 200), ("base", 200), ("ens_s1", 200)]:
    b = np.array([r["refit_auroc"] for r in rows if r["policy"] == p and r["n"] == n])
    d = load(p); po = d["split"] == "pool"; yp = d["y"][po]; yt = d["y"][:400]
    rng = np.random.default_rng(4242)
    v = np.array([auroc(yt, score(fit_lr(d["own"][po][i], yp[i]), d["own"][:400])) for i in [balanced_draw(yp, n, rng) for _ in range(300)]])
    m20 = np.array([rng.choice(v, 20, replace=False).mean() for _ in range(5000)])
    print(p, n, "build 20-draw mean %.4f | my 300 draws mean %.4f sd %.4f | share of my 20-draw means >= build: %.4f" % (b.mean(), v.mean(), v.std(), (m20 >= b.mean()).mean()))
