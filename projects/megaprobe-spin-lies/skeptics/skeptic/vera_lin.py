"""Own light arms (CPU, < 0.5 GB): 
  a  : layer-12 answer-token mean, StandardScaler + LR(C=0.05, balanced)            [1019 recipe]
  b  : final follow-up token ('Yes') at each of 24 layers, per-layer RMS scale on the training draw, LR(C=1, balanced)
       per layer, keep layers with train log-loss < 0.6 (all if none), score = mean of kept sigmoids  [Atlas/FIBS recipe]
  m24: 24 per-layer mean-pool LR probes (C=0.05), score = mean of their logits (a multi-layer linear ensemble on means)
  w  : TF-IDF words-only, same draws
Same protocol as vera_mega.py (own fold map FOLD_SEED, balanced n=400 draws, NDRAW draws, OOF mean). SHUFFLE=1 = null.
    cd kran-research && FOLD_SEED=77 NDRAW=5 uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_lin.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import MODELS, auroc, classes, load_answers, load_labels  # noqa: E402
from vera_io import CACHE, D20  # noqa: E402
from vera_mega import draw, fold_map  # noqa: E402  (my own helpers)

HERE = Path(__file__).resolve().parent
FOLD_SEED = int(os.environ.get("FOLD_SEED", 77)); NDRAW = int(os.environ.get("NDRAW", 5))
SHUFFLE = int(os.environ.get("SHUFFLE", 0)); SUFFIX = os.environ.get("READ_SUFFIX", "_own")
N = 1000


def run(model, fm):
    L = load_labels(model, D20)[:N]; c = classes(L); A = load_answers(model, D20)[:N]
    y = c["lie"].astype(int); keep = np.array([r["y_judge"] is not None for r in L])
    folds = np.array([fm[r["prompt_id"]] for r in L])
    meanL = np.load(HERE / "feats" / f"{model}{SUFFIX}.npz")["meanL"][:N].astype(np.float32)
    fin = np.asarray(np.load(CACHE / f"{model}{SUFFIX}" / "follow.npy", mmap_mode="r")[:N, -1], np.float32)  # (N,24,896)
    texts = [a["response"] for a in A]
    S = {k: np.zeros(N) for k in ["a", "b", "m24", "w"]}; nk = []
    for f in range(5):
        tr_all = np.flatnonzero((folds != f) & keep); te = np.flatnonzero(folds == f)
        for d in range(NDRAW):
            rng = np.random.default_rng(1000 * FOLD_SEED + 10 * f + d)
            idx = draw(tr_all, y, rng); yy = y.copy()
            if SHUFFLE: yy[idx] = rng.permutation(yy[idx])
            # a
            sc = StandardScaler().fit(meanL[idx, 11])
            lr = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(meanL[idx, 11]), yy[idx])
            S["a"][te] += lr.decision_function(sc.transform(meanL[te, 11])) / NDRAW
            # b
            probs, kept = [], 0
            for l in range(24):
                s = np.sqrt((fin[idx, l] ** 2).sum(1).mean())
                lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(fin[idx, l] / s, yy[idx])
                ll = log_loss(yy[idx], lr.predict_proba(fin[idx, l] / s)[:, 1])
                probs.append((ll, lr.predict_proba(fin[te, l] / s)[:, 1]))
            keepl = [p for ll, p in probs if ll < 0.6] or [p for _, p in probs]
            nk.append(len(keepl))
            S["b"][te] += np.mean(keepl, 0) / NDRAW
            # m24
            acc = np.zeros(len(te))
            for l in range(24):
                sc = StandardScaler().fit(meanL[idx, l])
                lr = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(meanL[idx, l]), yy[idx])
                acc += lr.decision_function(sc.transform(meanL[te, l])) / 24
            S["m24"][te] += acc / NDRAW
            # w
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            X = vec.fit_transform([texts[i] for i in idx])
            lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, yy[idx])
            S["w"][te] += lr.decision_function(vec.transform([texts[i] for i in te])) / NDRAW
    r = {"model": model, "b_layers_kept_mean": float(np.mean(nk))}
    for k, s in S.items():
        r[f"{k}_spin_fs"] = auroc(s[c["spin"]], s[c["truthful_fs"]]); r[f"{k}_spin_all"] = auroc(s[c["spin"]], s[c["truthful"]])
        r[f"{k}_lies"] = auroc(s[c["lie"]], s[c["truthful"]])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return r, S


if __name__ == "__main__":
    models = sys.argv[1:] or MODELS
    pids = set()
    for m in MODELS:
        pids |= {r["prompt_id"] for r in load_labels(m, D20)[:N]}
    fm = fold_map(pids, FOLD_SEED)
    tag = f"fs{FOLD_SEED}_d{NDRAW}_sh{SHUFFLE}{SUFFIX}"
    out = {}
    for m in models:
        r, S = run(m, fm)
        out[m] = dict(r, **{f"s_{k}": v.tolist() for k, v in S.items()})
        json.dump(out, open(HERE / f"vera_lin_{tag}.json", "w"))
