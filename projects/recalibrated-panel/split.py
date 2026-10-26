"""Poisoning vs score movement under Rule B (Iris, T5; frozen with PREREG before any scored run).

Rule B: each of the 20 members is re-standardised by the mean and sd (ddof 0) of a state's own 200 line-setting true
statements; the panel score is the max over members; the line is the 2nd-highest panel score of those 200.
Every quantity below is in Rule-B z units of its OWN state, so line and scores from two states can be compared.

2x2 split (catch of the 596 eval false statements):
  own        attacked false scores vs attacked line   (= the deployed Rule B catch)
  poison     CLEAN false scores    vs attacked line   (what the line move alone does)
  move       attacked false scores vs CLEAN line      (what the score move alone does; its false alarms reported)
Bound (research.md s5): after mean/sd standardisation of n values, the k-th highest is <= sqrt((n-k)/k); for the
2nd-highest of one member 9.95 (n = 200), for any single value sqrt(n-1) = 14.1. Reported per state: the line,
the bound it is measured against, and per member the top-2 z and the excess kurtosis of the 200 true z.

Head oracle line (Kasper #7766/#7772): the logistic and QDA heads of ceiling.py (same fit, probe+stop 800) with the
line at the 1% quantile of the 596 EVAL true statements (5th-highest; never fit, but the false-alarm set itself, so an
oracle), next to the frozen line (2nd-highest of the 200 refit trues). Tail-route label per seed: 'tail' if any
member's 2nd-highest line-set z exceeds the poisoning-only line that would cut clean false catch below 20%.

usage: python split.py STATE [STATE ...]   -> data/results/split.json (merged by state name)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CFG, DATA, threshold  # noqa: E402
from metrics import EVNEG, EVPOS, LINE  # noqa: E402
from ceiling import TRAIN, Y, load  # noqa: E402
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PANEL = CFG["panel"]["readouts"]
N = len(LINE)


def raw(st):
    for d in (DATA / "scores", DATA / "prior1022" / "scores"):
        f = d / f"{st}.npz"
        if f.exists():
            S = dict(np.load(f))
            return np.stack([S[f"tok{R}"] for R in PANEL], 1)
    raise FileNotFoundError(st)


def ruleb(X):
    mu, sd = X[LINE].mean(0), X[LINE].std(0)
    Zs = (X - mu) / sd
    return Zs, threshold(Zs.max(1)[LINE])


def kurt(z):
    return float(np.mean(z ** 4) - 3.0)  # z already has mean 0, var 1 on the line set


def heads_oracle(X):
    sc = StandardScaler().fit(X[TRAIN])
    out = {}
    for nm, h in [("logit", LogisticRegression(C=1.0, max_iter=5000)), ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1))]:
        s = h.fit(sc.transform(X[TRAIN]), Y[TRAIN]).decision_function(sc.transform(X))
        lo = threshold(s[EVNEG])  # 1% of 596 eval trues -> 5th-highest
        lf = threshold(s[LINE])
        out[nm] = {"oracle_line_catch": float((s[EVPOS] > lo).mean()), "oracle_line_fa": float((s[EVNEG] > lo).mean()),
                   "frozen_line_catch": float((s[EVPOS] > lf).mean()), "frozen_line_fa": float((s[EVNEG] > lf).mean())}
    return out


def tail_cut(Zc):
    """Smallest line at which the CLEAN false Rule-B max-z catch drops below 20%."""
    return float(np.quantile(Zc.max(1)[EVPOS], 0.80))


def main(states):
    Zc, lc = ruleb(raw("clean"))
    outf = DATA / "results" / "split.json"
    outf.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(outf.read_text()) if outf.exists() else {}
    for st in states:
        Z, la = ruleb(raw(st))
        fa, fc = Z.max(1)[EVPOS], Zc.max(1)[EVPOS]
        top2 = np.sort(Z[LINE], 0)[::-1][:2]  # (2, 20)
        r = {"line": la, "clean_line": lc, "bound_k2_one_member": float(np.sqrt((N - 2) / 2)),
             "bound_k1": float(np.sqrt(N - 1)),
             "own": float((fa > la).mean()), "poison": float((fc > la).mean()),
             "move": float((fa > lc).mean()), "move_false_alarm": float((Z.max(1)[EVNEG] > lc).mean()),
             "own_false_alarm": float((Z.max(1)[EVNEG] > la).mean()),
             "line_setter_members": [PANEL[int(j)] for j in Z[LINE][np.argsort(-Z[LINE].max(1))[:2]].argmax(1)],
             "false_max_z_median": float(np.median(fa)),
             "per_member": {str(R): {"top1": float(top2[0, j]), "top2": float(top2[1, j]), "kurt": kurt(Z[LINE, j])}
                            for j, R in enumerate(PANEL)}}
        r["heads_oracle"] = heads_oracle(load(st))
        r["tail_cut_20pct"] = tail_cut(Zc)
        r["route"] = "tail" if max(v["top2"] for v in r["per_member"].values()) > r["tail_cut_20pct"] else "gap"
        out[st] = r
        print(st, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_member",)},
              "max top2 member:", max(r["per_member"].items(), key=lambda kv: kv[1]["top2"])[0],
              "max kurt:", round(max(v["kurt"] for v in r["per_member"].values()), 1))
    outf.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
