"""Post-hoc row 'not attacked' (Tove #7815; exploratory, never the status): the logistic and QDA heads of ceiling.py
(same settings) fit ONCE on the CLEAN model's labelled probe+stop scores (labels a deployer has), then applied
unchanged to every state; the line is re-set on that state's own 200 line-setting trues (2nd-highest).
Also a 'difference' rule with no labels on any model: each member minus the mean over members (removes one offset
shared by all 20 members), re-standardised on the state's 200 trues, max, one line (the label-free counterpart).
usage: python clean_head.py STATE [...]  -> data/results/clean_head.json (merged by state name)
"""
import json
import sys

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ceiling import TRAIN, Y, load
from common import DATA, threshold
from metrics import EVNEG, EVPOS, LINE


def catch(s):
    t = threshold(s[LINE])
    return [float((s[EVPOS] > t).mean()), float((s[EVNEG] > t).mean())]


def main(states):
    Xc = load("clean")
    sc = StandardScaler().fit(Xc[TRAIN])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xc[TRAIN]), Y[TRAIN])
    q = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(sc.transform(Xc[TRAIN]), Y[TRAIN])
    f = DATA / "results" / "clean_head.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for st in states:
        X = load(st)
        D = X - X.mean(1, keepdims=True)  # remove the offset shared by all members
        Db = ((D - D[LINE].mean(0)) / D[LINE].std(0)).max(1)
        out[st] = {"logit_cleanfit": catch(lr.decision_function(sc.transform(X))),
                   "qda_cleanfit": catch(q.decision_function(sc.transform(X))),
                   "diff_ruleB": catch(Db)}
        print(st, out[st], flush=True)
    f.write_text(json.dumps(out, indent=1))
    # Tove #7829 / Drew #7826: the same two keys merged into ceiling.json (run AFTER ceiling.py, which rewrites a state)
    cf = DATA / "results" / "ceiling.json"
    ce = json.loads(cf.read_text())
    for st in states:
        if st in ce:
            ce[st]["logit_cleanfit"], ce[st]["qda_cleanfit"] = out[st]["logit_cleanfit"], out[st]["qda_cleanfit"]
    cf.write_text(json.dumps(ce, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
