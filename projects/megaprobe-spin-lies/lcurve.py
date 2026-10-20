"""Label-count curve and probe size (Esben, 2026-10-20 11:43: "is it just overfitting because the dataset is small?
Maybe try more probe sizes").

Per model and outer fold (same folds as the main run), fit the one-layer probe (arm a) and the megaprobe (arm c) on
n = 100, 200, 400 and ALL labelled training answers (~600-800). The 400 set is draw 0 of the main run; 200 and 100 are
nested class-balanced subsets of it; ALL is every labelled answer in the four training folds (class-weighted loss).
Probe size: the megaprobe at 32 principal components (top 32 of the same 128, so the same whitening) at n = 400 and
ALL. One draw per size, so read differences under ~0.03 as noise. Hyperparameters as in the main run (data/hp.json).
Fixed before any result of this script was seen (room post #3653).

    uv run python projects/megaprobe-spin-lies/lcurve.py <policy> [...]  -> data/lcurve/<policy>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes as P  # noqa: E402
from convergence import subsets  # noqa: E402

OUTD = P.DATA / "lcurve"
SIZES = ["100", "200", "400", "all"]
SUBS = ("spin_vs_fact_truthful", "spin_vs_all_truthful", "lies_vs_truthful")


def balanced_subset(tr, y, n, rng):
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    n_neg = min(len(neg), max(n // 2, n - len(pos)))
    n_pos = min(len(pos), n - n_neg)
    return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]))


def run(p):
    OUTD.mkdir(parents=True, exist_ok=True)
    fm = P.folds(); rows = P.load_rows(p); D = P.draws_for(p, rows, fm); y = P.yvec(rows)
    fold = np.array([fm[r["prompt_id"]] for r in rows])
    cache = P.Cache(p, "own"); X12 = cache.mean_layer(12)
    arms = [f"a_{s}" for s in SIZES] + [f"c_{s}" for s in SIZES] + ["c32_400", "c32_all"]
    sc = {a: np.full(P.N_ROWS, np.nan) for a in arms}
    sizes = {}
    t0 = time.time()
    for k in range(P.N_FOLDS):
        te = np.where(fold == k)[0]
        rng = np.random.default_rng([P.SEED, P.POLICIES.index(p), k, 300])
        d400 = np.array(D[k][0])
        d200 = balanced_subset(d400, y, 200, rng)
        d100 = balanced_subset(d200, y, 100, rng)
        dall = np.array([i for i in range(P.N_ROWS) if fold[i] != k and y[i] >= 0])
        tr = {"100": d100, "200": d200, "400": d400, "all": dall}
        sizes[k] = {s: [int((y[v] == 0).sum()), int((y[v] == 1).sum())] for s, v in tr.items()}
        pca = P.pca_fit([(cache, [i for i in range(P.N_ROWS) if fold[i] != k])], P.HP["k"])
        Z, offs = P.pca_project(cache, pca)
        seed = P.SEED + 100 * k
        for s in SIZES:
            sc[f"a_{s}"][te] = P.fit_a(X12, y, tr[s], te)
            sc[f"c_{s}"][te] = P.fit_c(Z, offs, y, tr[s].tolist(), te, seed)[0]
        Z32 = Z[:, :, -32:].contiguous()  # top 32 components (eigh is ascending)
        del Z; torch.mps.empty_cache()
        for s in ("400", "all"):
            sc[f"c32_{s}"][te] = P.fit_c(Z32, offs, y, tr[s].tolist(), te, seed)[0]
        del Z32; torch.mps.empty_cache()
        print(f"{p} fold {k} done {time.time() - t0:.0f}s", flush=True)
    S = subsets(rows)
    res = {a: {n: round(float(roc_auc_score(S[n][1][S[n][0]], v[S[n][0]])), 4) for n in SUBS} for a, v in sc.items()}
    (OUTD / f"{p}.json").write_text(json.dumps({"policy": p, "hp": P.HP, "sizes": sizes, "auroc": res,
                                                "scores": {a: [round(float(x), 5) for x in v] for a, v in sc.items()},
                                                "seconds": round(time.time() - t0)}, indent=1))
    for a, r in res.items():
        print(p, a, json.dumps(r), flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
