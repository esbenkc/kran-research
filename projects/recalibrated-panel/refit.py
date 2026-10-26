"""PREREG s8 re-fit recovery on attacked states. For each state: re-fit the country-token probe tok_R (R = 10..29) on THAT
state's clean-training-pool activations (probe+attack+stop, 1400 statements, 350 cities), C by the 1020/1021 rule
(5-fold GroupKFold by city, grid {0.01, 0.1, 1, 10}, out-of-fold AUROC, ties within 0.002 -> smaller C), line from
that state's 200 line-set true statements. Best readout = highest training-pool OOF AUROC (no eval data; ties ->
lower R). Convergence: lbfgs n_iter_ < max_iter (5000) per fit. Label-count curve at the best readout: fits on 50 / 100
/ 200 / 350 training-pool cities (= 200 / 400 / 800 / 1400 labels; nested city subsets, seed 1022), C fixed at the
best readout's C. Shuffled-label null: 5 permutations of the training-pool labels at the best readout.
-> data/refit/<state>.npz (eval-ready scores per readout + curve + null), data/refit/<state>.json,
   and with --summary: data/results/refit.json (pooled per arm).
Run: .venv/bin/python refit.py <state> [<state> ...] ; .venv/bin/python refit.py --summary
"""
import json
import sys
import time

import numpy as np

from common import CFG, DATA, Acts, pool_idx, rows
from metrics import boot_catch, catch
from panel import FITS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

OUT = DATA / "refit"
READ = CFG["panel"]["readouts"]
GRID = [0.01, 0.1, 1.0, 10.0]
TIE = 0.002
CURVE = [50, 100, 200, 350]


def fit(X, y, tr, C):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return sc, m


def folds(idx, city, k=5):
    """1020 rule: GroupKFold by city, groups shuffled with seed 1020 first."""
    cs = sorted(set(city[idx]))
    perm = np.random.default_rng(1020).permutation(len(cs))
    gid = {cs[j]: i for i, j in enumerate(perm)}
    g = np.array([gid[c] for c in city[idx]])
    return [(idx[a], idx[b]) for a, b in GroupKFold(k).split(idx, groups=g)]


def select(X, y, city, tr):
    res, conv = {}, True
    for C in GRID:
        oof = np.zeros(len(y))
        for a, b in folds(tr, city):
            sc, m = fit(X, y, a, C)
            conv &= bool(m.n_iter_[0] < 5000)
            oof[b] = m.decision_function(sc.transform(X[b]))
        res[C] = float(roc_auc_score(y[tr], oof[tr]))
    best = max(res.values())
    return min(c for c in GRID if res[c] >= best - TIE), res, conv


def one(state):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _, y, city, _ = rows()
    tr, _, _ = pool_idx()
    trows = np.load(FITS / "panel.npz")["tok_rows_clean"]
    A = Acts(state)
    S, info = {}, {}
    for R in READ:
        X = A.at(R, trows)
        C, res, cv_conv = select(X, y, city, tr)
        sc, m = fit(X, y, tr, C)
        S[f"tok{R}"] = m.decision_function(sc.transform(X))
        info[R] = {"C": C, "oof_auroc": res[C], "grid": res, "n_iter": int(m.n_iter_[0]),
                   "converged": bool(m.n_iter_[0] < 5000), "cv_fits_converged": cv_conv,
                   "eval_catch": catch(S[f"tok{R}"])["catch"]}
        print(state, R, C, round(res[C], 4), info[R]["eval_catch"], f"{time.time() - t0:.0f}s", flush=True)
    best = max(READ, key=lambda r: (info[r]["oof_auroc"], -r))
    X = A.at(best, trows)
    cs = sorted(set(city[tr]))
    order = np.random.default_rng(1022).permutation(len(cs))
    curve = {}
    for k in CURVE:
        keep = set(cs[j] for j in order[:k])
        sub = tr[np.isin(city[tr], list(keep))]
        sc, m = fit(X, y, sub, info[best]["C"])
        S[f"curve{k}"] = m.decision_function(sc.transform(X))
        curve[k] = {"n_labels": int(len(sub)), "eval_catch": catch(S[f"curve{k}"])["catch"],
                    "converged": bool(m.n_iter_[0] < 5000)}
    null = []
    for p in range(1, 6):
        yp = y.copy()
        yp[tr] = y[tr][np.random.default_rng(p).permutation(len(tr))]
        sc, m = fit(X, yp, tr, info[best]["C"])
        S[f"null{p}"] = m.decision_function(sc.transform(X))
        null.append(catch(S[f"null{p}"])["catch"])
    np.savez(OUT / f"{state}.npz", **S)
    (OUT / f"{state}.json").write_text(json.dumps({"state": state, "best_readout": best, "per_readout": info,
                                                   "label_curve": curve, "null_catch_at_best": null,
                                                   "seconds": round(time.time() - t0, 1)}, indent=1))
    print(state, "best", best, info[best]["eval_catch"], curve, null, f"{time.time() - t0:.0f}s", flush=True)


def summary():
    arms = {"panel": [f"panel__lam10000__seed{s}" for s in range(3)], "panel_lam0": ["panel__lam0__seed0"],
            "panel_lam1000": ["panel__lam1000__seed0"]}
    out = {}
    for a, sts in arms.items():
        sts = [s for s in sts if (OUT / f"{s}.json").exists()]
        if not sts:
            continue
        J = {s: json.loads((OUT / f"{s}.json").read_text()) for s in sts}
        Z = {s: np.load(OUT / f"{s}.npz") for s in sts}
        rec = boot_catch([Z[s][f"tok{J[s]['best_readout']}"] for s in sts], seed=1022)
        out[a] = {"states": sts, "best_readout_per_seed": {s: J[s]["best_readout"] for s in sts},
                  "recovery_at_best": rec,
                  "per_readout_pooled": {R: float(np.mean([J[s]["per_readout"][str(R)]["eval_catch"] for s in sts]))
                                         for R in READ},
                  "all_fits_converged": all(v["converged"] and v["cv_fits_converged"]
                                            for s in sts for v in J[s]["per_readout"].values()),
                  "label_curve_pooled": {k: float(np.mean([J[s]["label_curve"][str(k)]["eval_catch"] for s in sts]))
                                         for k in CURVE},
                  "null_pooled": float(np.mean([np.mean(J[s]["null_catch_at_best"]) for s in sts]))}
    (DATA / "results" / "refit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary()
    else:
        for st in sys.argv[1:]:
            one(st)
