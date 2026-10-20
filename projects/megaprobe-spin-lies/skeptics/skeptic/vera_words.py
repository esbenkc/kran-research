"""Words-only floor (own code): TF-IDF (1-2 grams) + logistic regression on the answer text.
Protocol mirrors the frozen plan: 5 folds grouped by prompt_id, fit on a class-balanced draw of n=400 (lie vs honest,
all labels) from the other 4 folds, out-of-fold score for every row. Also a fit on fact-stated rows only
(spin vs truthful_fs) as the diagnostic arm. Reports spin-vs-truthful AUROC per model and mean over the 4 trained.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_words.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import MODELS, TRAINED, auroc, classes, load_answers, load_labels

SEED = 7
N_FIT = 400


def fold_map(pids, k=5, seed=SEED):
    u = sorted(set(pids)); rng = np.random.default_rng(seed); rng.shuffle(u)
    return {p: i % k for i, p in enumerate(u)}


def oof(texts, y, folds, fitmask, rng, n_fit=N_FIT):
    s = np.full(len(y), np.nan)
    for f in range(5):
        tr = np.flatnonzero((folds != f) & fitmask); te = np.flatnonzero(folds == f)
        pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]; h = min(n_fit // 2, len(pos), len(neg))
        pick = np.concatenate([rng.choice(pos, h, replace=False), rng.choice(neg, h, replace=False)])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform([texts[i] for i in pick])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y[pick])
        s[te] = clf.decision_function(vec.transform([texts[i] for i in te]))
    return s


res = {}
allp = set()
for m in MODELS:
    allp |= {a["prompt_id"] for a in load_answers(m)}
fm = fold_map(allp)
for m in MODELS:
    A = load_answers(m); L = load_labels(m)
    assert [a["prompt_id"] for a in A] == [r["prompt_id"] for r in L]
    c = classes(L); texts = [a["response"] for a in A]
    keep = np.array([r["y_judge"] is not None for r in L])
    y = c["lie"].astype(int); folds = np.array([fm[a["prompt_id"]] for a in A])
    rng = np.random.default_rng(SEED)
    s_all = oof(texts, y, folds, keep, rng)
    fs = c["spin"] | c["truthful_fs"]
    s_fs = oof(texts, y, folds, fs, rng, n_fit=min(N_FIT, 2 * min(c["spin"].sum(), c["truthful_fs"].sum()) * 4 // 5))
    r = dict(n_spin=int(c["spin"].sum()), n_truth=int(c["truthful"].sum()), n_truth_fs=int(c["truthful_fs"].sum()),
             all_lies=auroc(s_all[c["lie"]], s_all[c["truthful"]]),
             spin_vs_truth=auroc(s_all[c["spin"]], s_all[c["truthful"]]),
             spin_vs_truth_fs=auroc(s_all[c["spin"]], s_all[c["truthful_fs"]]),
             fsfit_spin_vs_truth_fs=auroc(s_fs[c["spin"]], s_fs[c["truthful_fs"]]))
    res[m] = r
    print(m, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
print("mean over 4 trained spin_vs_truth %.3f | fs-fit %.3f" % (
    np.mean([res[m]["spin_vs_truth"] for m in TRAINED]), np.mean([res[m]["fsfit_spin_vs_truth_fs"] for m in TRAINED])))
json.dump(res, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
