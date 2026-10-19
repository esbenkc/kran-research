"""Hugo's caveat, rebuilt with my own code: train the n=200 refit only on pool answers where judge and string check
agree (clean labels), score on the 400 test answers with (a) all judge labels, (b) the clean test subset.
Gap = base - policy, paired bootstrap over test prompts (1000, seed 1019). 3 draw seeds x 20 draws.
    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_clean_train.py"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, balanced_draw, fit_lr, score
from vera_rebuild import LAM1, load
data = {p: load(p) for p in ["base"] + LAM1}
boots = np.random.default_rng(1019).integers(0, 400, (1000, 400))
out = {}
for seed in (71, 72, 73):
    rng = np.random.default_rng(seed); S = {}; Y = {}
    for p, d in data.items():
        po = np.flatnonzero(d["split"] == "pool"); ypo = d["y"][po]
        cl = (ypo >= 0) & (ypo == d["ys"][po])
        pc, yc = po[cl], ypo[cl]
        S[p] = np.array([score(fit_lr(d["own"][pc[i]], yc[i]), d["own"][:400]) for i in [balanced_draw(yc, 200, rng) for _ in range(20)]])
        yt = d["y"][:400]; Y[p] = {"full": yt, "clean": np.where((yt >= 0) & (yt == d["ys"][:400]), yt, -1)}
        if seed == 71: print(p, "clean pool honest/deceptive", int((yc == 0).sum()), int((yc == 1).sum()))
    for lab in ("full", "clean"):
        for p in LAM1:
            f = lambda ix: np.mean([auroc(Y["base"][lab][ix], s[ix]) for s in S["base"]]) - np.mean([auroc(Y[p][lab][ix], s[ix]) for s in S[p]])
            g = np.array([f(ix) for ix in boots]); pt = f(np.arange(400)); lo, hi = np.percentile(g, [2.5, 97.5])
            out[f"{seed}:{lab}:{p}"] = [pt, lo, hi]
            print(f"seed {seed} test={lab:5s} {p}: gap {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  base refit {np.mean([auroc(Y['base'][lab], s) for s in S['base']]):.3f}")
json.dump(out, open(Path(__file__).resolve().parent / "vera_clean_train.json", "w"), indent=1)
