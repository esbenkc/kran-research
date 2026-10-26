"""Aggregation ceilings (PREREG s5), frozen with PREREG.md before any scored run.

Method = Iris's Phase-0 check `agg_ceiling_1022.py` (room #7742), unchanged: only the paths and the state list moved
so it runs in this project on tonight's states and on the saved theory-1022 states (data/prior1022/scores).

Per state, catch of the 596 eval false statements (and false alarms on the 596 eval true) at a line that 1% of true
statements cross (2nd-highest of the state's own 200 line-setting true statements):
  plain    max of clean-standardised z, own line (1022 headline rule)
  cleanln  same max, line from the CLEAN model (false alarms reported)
  ruleB    each member re-standardised on the state's own 200 line-setting trues (mean, sd ddof 0), then max, own line
  best1    best single member at its own line (a lower bound on the best per-member-rescaled max rule)
  logit    LogisticRegression(C=1.0) on the 20 member z-scores, fit WITH labels on the state's probe+stop (800)
  logit_shuffled  the same head on shuffled labels (null)
  qda      QuadraticDiscriminantAnalysis(reg_param=0.1), same fit set
Splits are city-disjoint, so fit, line and eval use different cities.

usage: python ceiling.py STATE [STATE ...]   (a state is looked up in data/scores/, then data/prior1022/scores/)
       writes data/results/ceiling.json (merged by state name)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CFG, DATA, rows, threshold  # noqa: E402
from metrics import EVNEG, EVPOS, LINE  # noqa: E402

PANEL = CFG["panel"]["readouts"]
Z = json.loads((DATA / "fits" / "zmoments.json").read_text())
_R, Y, _C, SPLIT = rows()
TRAIN = np.where(np.isin(SPLIT, ["probe", "stop"]))[0]


def zmat(S):
    return np.stack([(S[f"tok{R}"] - Z[f"tok{R}"]["mu"]) / Z[f"tok{R}"]["sd"] for R in PANEL], 1)


def catch(s, line):
    return float((s[EVPOS] > line).mean()), float((s[EVNEG] > line).mean())


def load(st):
    for d in (DATA / "scores", DATA / "prior1022" / "scores"):
        f = d / f"{st}.npz"
        if f.exists():
            return zmat(dict(np.load(f)))
    raise FileNotFoundError(st)


def ceilings(X, clean_line):
    r = {}
    m = X.max(1)
    r["plain"] = catch(m, threshold(m[LINE]))
    r["cleanln"] = catch(m, clean_line)
    mu, sd = X[LINE].mean(0), X[LINE].std(0)
    mb = ((X - mu) / sd).max(1)
    r["ruleB"] = catch(mb, threshold(mb[LINE]))
    per = [catch(X[:, j], threshold(X[LINE, j]))[0] for j in range(X.shape[1])]
    j = int(np.argmax(per))
    r["best1"] = (per[j], PANEL[j])
    sc = StandardScaler().fit(X[TRAIN])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[TRAIN]), Y[TRAIN])
    s = lr.decision_function(sc.transform(X))
    r["logit"] = catch(s, threshold(s[LINE]))
    ysh = np.random.default_rng(0).permutation(Y[TRAIN])  # null: same head, shuffled labels
    lr0 = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[TRAIN]), ysh)
    s = lr0.decision_function(sc.transform(X))
    r["logit_shuffled"] = catch(s, threshold(s[LINE]))
    q = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(sc.transform(X[TRAIN]), Y[TRAIN])
    s = q.decision_function(sc.transform(X))
    r["qda"] = catch(s, threshold(s[LINE]))
    return r


def main(states):
    clean_line = threshold(load("clean").max(1)[LINE])
    outf = DATA / "results" / "ceiling.json"
    outf.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(outf.read_text()) if outf.exists() else {}
    for st in states:
        r = ceilings(load(st), clean_line)
        out[st] = r
        print(st, {k: tuple(round(x, 3) if isinstance(x, float) else x for x in v) for k, v in r.items()})
    outf.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
