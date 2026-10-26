"""Mechanism diagnostic (Iris): is the spread the in-loop evader adds to true statements COMMON across the 20 members
(one statement-level offset at every layer; a labelled linear head can subtract it) or member-specific?

Per state, on the 200 line-set + 600 CAL600 true statements (never in the attack loss):
  pc1_share       top eigenvalue / trace of the 20x20 correlation of the members' raw true scores
  dprime_median   median over members of (mean false - mean true) / sd true on eval (Cohen's d, per member)
  dprime_max      the largest member d' and its readout
  sd_ratio_med    median over members of sd_true(state) / sd_true(clean)
  maha            whitened separation Delta = sqrt(delta' S^-1 delta), delta = class-mean difference, S = pooled
                  within-class covariance, both from the probe+stop TRAIN rows (the heads' fit set)
  lda_pred_catch  Gaussian equal-covariance prediction of the best linear head's catch at a 1% line:
                  Phi(Delta - z_0.99), z_0.99 = 2.326 (Neyman-Pearson for two Gaussians with shared covariance)
  diag_sep        sqrt(sum_R d'_R^2) on TRAIN, the separation if members were independent (what the loss sees)
usage: python commonmode.py STATE [STATE ...]   (looks in the project data/scores, then data/prior1022/scores)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
from common import CFG, DATA, rows  # noqa: E402
from metrics import EVNEG, EVPOS, LINE  # noqa: E402

PANEL = CFG["panel"]["readouts"]
_R, Y, _C, SPLIT = rows()
TRUE = np.where(np.isin(SPLIT, ["probe", "stop", "refit"]) & (Y == 0))[0]


def raw(st):
    for d in (DATA / "scores", DATA / "prior1022" / "scores"):
        f = d / f"{st}.npz"
        if f.exists():
            S = dict(np.load(f))
            return np.stack([S[f"tok{R}"] for R in PANEL], 1)
    raise FileNotFoundError(st)


C0 = raw("clean")
out = {}
for st in sys.argv[1:]:
    X = raw(st)
    ev = np.linalg.eigvalsh(np.corrcoef(X[TRUE].T))
    dp = (X[EVPOS].mean(0) - X[EVNEG].mean(0)) / X[EVNEG].std(0)
    j = int(np.argmax(dp))
    r = {"pc1_share": float(ev[-1] / ev.sum()), "dprime_median": float(np.median(dp)),
         "dprime_max": [float(dp[j]), PANEL[j]],
         "sd_ratio_med": float(np.median(X[TRUE].std(0) / C0[TRUE].std(0)))}
    TR = np.where(np.isin(SPLIT, ["probe", "stop"]))[0]
    A, B = X[TR][Y[TR] == 1], X[TR][Y[TR] == 0]
    dl = A.mean(0) - B.mean(0)
    S = 0.5 * (np.cov(A.T) + np.cov(B.T))
    D = float(np.sqrt(dl @ np.linalg.solve(S, dl)))
    r |= {"maha": D, "lda_pred_catch": float(norm.cdf(D - norm.ppf(0.99))),
          "diag_sep": float(np.sqrt(np.sum(dl ** 2 / np.diag(S))))}
    out[st] = r
    print(st, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
f = Path(__file__).with_suffix(".json")
old = json.loads(f.read_text()) if f.exists() else {}
f.write_text(json.dumps(old | out, indent=1))
