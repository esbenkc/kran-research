"""Hugo (skeptic2): which side moves when the refit is trained on clean labels only?
Clean = judge label equals string check. Scored on the clean TEST subset, n=200, 20 draws, seed 11."""
import numpy as np
from sklearn.metrics import roc_auc_score
from check import load, fit_score

for p in ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]:
    d = load(p)
    te = np.arange(400)
    y = d["y"][te]
    m = ~np.isnan(y) & (d["ys"][te] == y)
    yy = y[m].astype(int)
    row = [p]
    for clean_train in [False, True]:
        rng = np.random.default_rng(11)
        pool = np.where(d["split"] == "pool")[0]
        ok = ~np.isnan(d["y"][pool])
        if clean_train:
            ok &= d["ys"][pool] == d["y"][pool]
        pool = pool[ok]
        h, dd = pool[d["y"][pool] == 0], pool[d["y"][pool] == 1]
        A = []
        for _ in range(20):
            f = np.concatenate([rng.choice(h, 100, replace=False), rng.choice(dd, 100, replace=False)])
            s, _ = fit_score(d["own"][f], d["y"][f].astype(int), d["own"][te])
            A.append(roc_auc_score(yy, s[m]))
        row += [("clean" if clean_train else "judge") + "-train", round(float(np.mean(A)), 3), f"pool h/d {len(h)}/{len(dd)}"]
    print(*row, f"clean test h/d {int((yy == 0).sum())}/{int(yy.sum())}")
