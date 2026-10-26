"""Vera: stress test of the 'no rule on the 20 scores' wording (exploratory, NOT the status).
The frozen heads are logistic C=1 and QDA reg 0.1. A stronger head that reaches >= 20% at <= 2% eval false alarms
would make 'every rule' a stretched definition. Heads (all fit on the state's 800 probe+stop, line = 2nd-highest
of the 200 line-set trues, eval 596/596):
  logistic C in {0.01, 0.1, 1, 10, 100}; QDA reg in {0, 0.01, 0.1, 0.5}; HistGradientBoosting; kNN(15);
  MLP(64,64); logistic on pairwise products (degree-2 poly).
Also: an 'upper-bound' head trained WITH the eval labels (in-sample, 5-fold by city on eval) to see what is left.
Usage: python vera_strong.py <state>|--scores-npz PATH  [tag]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vera26 as V  # own module

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def score_fn(clf, X):
    if hasattr(clf, "decision_function"):
        try:
            return clf.decision_function(X)
        except Exception:
            pass
    p = clf.predict_proba(X)[:, 1]
    return np.log(p + 1e-12) - np.log(1 - p + 1e-12) + 1e-9 * np.arange(len(p)) * 0


def models():
    m = {}
    for C in (0.01, 0.1, 1, 10, 100):
        m[f"logit_C{C}"] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    for r in (0.0, 0.01, 0.1, 0.5):
        m[f"qda_r{r}"] = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=r))
    m["hgb"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    m["knn15"] = make_pipeline(StandardScaler(), KNeighborsClassifier(15))
    m["mlp"] = make_pipeline(StandardScaler(), MLPClassifier((64, 64), max_iter=2000, random_state=0, alpha=1e-3))
    m["poly2_logit"] = make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(),
                                     LogisticRegression(C=0.1, max_iter=5000))
    return m


def run(S):
    out = {}
    Xtr, ytr = S[V.TRAIN], V.YF[V.TRAIN]
    for name, clf in models().items():
        clf.fit(Xtr, ytr)
        h = score_fn(clf, S)
        # ties at the line (kNN, hgb) -> strict '>' already conservative for catch
        c = V.catch([V.flag(h)])
        out[name] = dict(catch=round(c["catch"], 4), lo=round(c["lo"], 4), hi=round(c["hi"], 4), fa=round(c["fa"][0], 4),
                         auroc=round(V.auroc(h[V.EVAL], V.YF[V.EVAL]), 4))
    # eval-label upper bound: 5-fold by city inside eval, logistic C=1 and QDA 0.1; line = 2nd highest of line-set trues
    ev = np.where(V.EVAL)[0]
    for name in ("logit_C1", "qda_r0.1"):
        hs = np.full(len(S), np.nan)
        for tr, te in GroupKFold(5).split(ev, groups=V.CITY[ev]):
            clf = models()[name].fit(S[ev[tr]], V.YF[ev[tr]])
            hs[ev[te]] = score_fn(clf, S[ev[te]])
        clf = models()[name].fit(S[ev], V.YF[ev])
        hs[V.LINE_T] = score_fn(clf, S[V.LINE_T])
        t = np.sort(hs[V.LINE_T])[::-1][1]
        out["evalCV_" + name] = dict(catch=round(float((hs[V.EVF] > t).mean()), 4), fa=round(float((hs[V.EVT] > t).mean()), 4))
    return out


if __name__ == "__main__":
    a = sys.argv[1]
    S = V.scores_from_npz(sys.argv[2]) if a == "--scores-npz" else V.scores_from_ctok(a)
    tag = sys.argv[3] if a == "--scores-npz" and len(sys.argv) > 3 else (sys.argv[2] if a != "--scores-npz" and len(sys.argv) > 2 else a)
    res = run(S)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"vera_strong_{tag}.json"), "w"), indent=1)
    for k, v in res.items():
        print(tag, k, v)
