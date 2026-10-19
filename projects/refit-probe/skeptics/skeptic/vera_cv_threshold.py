"""Rebuild of build's D4 deployable line (own code): threshold from 5-fold out-of-fold scores inside the n=200 fit
answers, k-th highest out-of-fold HONEST score, k = max(1, floor(0.05 (n_honest + 1))); the final probe (fit on all
200) is applied to the 400 test answers. Reports realised test false alarms and catch. 20 draws, seed 57.
    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_cv_threshold.py"""
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import balanced_draw, fit_lr, score
from vera_rebuild import load
rng = np.random.default_rng(57)
for p in ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]:
    d = load(p); te, po = d["split"] == "test", d["split"] == "pool"
    yt, yp, Xp, Xt = d["y"][te], d["y"][po], d["own"][po], d["own"][te]
    fa, ca, fa_in = [], [], []
    for k in range(20):
        idx = balanced_draw(yp, 200, rng); X, y = Xp[idx], yp[idx]
        oof = np.empty(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=k).split(X, y):
            oof[va] = score(fit_lr(X[tr], y[tr]), X[va])
        m = fit_lr(X, y); st = score(m, Xt); sin = score(m, X)
        kk = max(1, int(np.floor(0.05 * ((y == 0).sum() + 1))))
        thr = np.sort(oof[y == 0])[::-1][kk - 1]
        thr_in = np.sort(sin[y == 0])[::-1][kk - 1]
        fa.append((st[yt == 0] > thr).mean()); ca.append((st[yt == 1] > thr).mean()); fa_in.append((st[yt == 0] > thr_in).mean())
    print(p, "CV line: test false alarm %.3f catch %.3f | in-sample line false alarm %.3f" % (np.mean(fa), np.mean(ca), np.mean(fa_in)))
