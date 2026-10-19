"""Rebuild build's per-draw refit AUROC from its saved fit indices (results/draws.jsonl) with my own fit + AUROC.
    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_exact_draws.py"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, fit_lr, score
from vera_rebuild import load
rows = [json.loads(l) for l in open(__import__("vera_rebuild").D / "results/draws.jsonl")]
for p, n in [("lam1_s0", 400), ("lam1_s0", 200), ("ens_s1", 200), ("base", 200)]:
    d = load(p); yt = d["y"][:400]
    R = [r for r in rows if r["policy"] == p and r["n"] == n]
    mine = [auroc(yt, score(fit_lr(d["own"][np.array(r["fit_idx"])], d["y"][np.array(r["fit_idx"])]), d["own"][:400])) for r in R]
    theirs = [r["refit_auroc"] for r in R]
    print(p, n, "draws", len(R), "my mean %.4f build mean %.4f max|diff| %.2e" % (np.mean(mine), np.mean(theirs), np.max(np.abs(np.array(mine) - theirs))))
