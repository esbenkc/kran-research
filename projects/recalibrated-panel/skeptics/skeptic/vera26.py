"""Vera (skeptic), theory-1026-recalibrated-panel. Own rebuild from RAW data; never imports build's code.

Definitions (PREREG.md v1.1, s3/s5/s7), written from the text, not from analysis.py/ceiling.py:
  score   s_R(x) = ((h_R(x) - mu_R)/sd_R) . w_R + b_R, panel.npz, readouts 10..29, positive = false
  line    2nd-highest of the scored state's own 200 line-setting ('refit') TRUE statements; flag iff score > line
  Rule B  per member z = (s - mean)/sd (ddof 0) with mean/sd from the state's own 200 line-set trues; max over 20
  plain   clean-model z-moments (zmoments.json, fit-pool trues), max, own line
  clean-model rule  clean moments + the CLEAN model's line, applied to the attacked state
  heads   20 raw scores -> StandardScaler (fit on train) -> Logistic(C=1, max_iter 5000) | QDA(reg 0.1);
          train = state's probe+stop (800), line on the 200 line-set trues, eval 596 F / 596 T
  CI      city-cluster bootstrap over eval cities (own seed 26, 2000 draws), lines fixed

Usage: python vera26.py <state> [--scores-npz PATH]   (default: score from data/ctok/<state>.npy)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
_IN_REPO = (HERE.parents[1] / "PREREG.md").exists()  # receipts copy: projects/recalibrated-panel/skeptics/skeptic
PROJ = Path(os.environ.get("VERA_PROJ", HERE.parents[1] if _IN_REPO else HERE.parents[1] / "kran-research/projects/recalibrated-panel"))
D = PROJ / "data"
ROWS = json.load(open(D / "ref/split.json"))
YF = np.array([1 - r["true"] for r in ROWS])
SPLIT = np.array([r["split"] for r in ROWS])
CITY = np.array([r["city"] for r in ROWS])
TRAIN = np.isin(SPLIT, ["probe", "stop"])
LINE_T = (SPLIT == "refit") & (YF == 0)
CAL600 = np.isin(SPLIT, ["probe", "stop", "refit"]) & (YF == 0)
EVAL = SPLIT == "eval"
EVF = np.where(EVAL & (YF == 1))[0]
EVT = np.where(EVAL & (YF == 0))[0]
EV_CITIES = np.array(sorted(set(CITY[EVAL])))
EVF_CID = np.searchsorted(EV_CITIES, CITY[EVF])
EVT_CID = np.searchsorted(EV_CITIES, CITY[EVT])
assert len(EVF) == 596 and len(EVT) == 596 and LINE_T.sum() == 200 and TRAIN.sum() == 800 and CAL600.sum() == 600
R = list(range(10, 30))
PANEL = np.load(D / "fits/panel.npz")
ZM = json.load(open(D / "fits/zmoments.json"))


def city_W(n_boot=2000, seed=26):
    rng = np.random.default_rng(seed)
    nc = len(EV_CITIES)
    return np.stack([np.bincount(rng.integers(0, nc, nc), minlength=nc) for _ in range(n_boot)]).astype(float)


W = city_W()


def scores_from_ctok(state):
    X = np.load(D / f"ctok/{state}.npy", mmap_mode="r")  # 2992 x 20 x 1536 fp16
    assert X.shape[1] == 20, X.shape
    S = np.empty((X.shape[0], 20))
    for j, r in enumerate(R):
        k = f"tok{r}"
        h = np.asarray(X[:, j, :], dtype=np.float64)
        S[:, j] = ((h - PANEL[k + "__mu"]) / PANEL[k + "__sd"]) @ PANEL[k + "__w"] + float(PANEL[k + "__b"])
    return S


def scores_from_npz(path):
    z = np.load(path)
    return np.stack([z[f"tok{r}"] for r in R], 1).astype(np.float64)


def line_of(s, idx=LINE_T, k=2):
    return np.sort(s[idx])[::-1][k - 1]


def catch(flag_all):
    """pooled catch over states (list of 2992-bool arrays) with city-cluster CI and eval false alarms."""
    fl = np.zeros(len(EV_CITIES)); nf = np.zeros(len(EV_CITIES)); fa = []
    for f in flag_all:
        np.add.at(fl, EVF_CID, f[EVF]); np.add.at(nf, EVF_CID, 1); fa.append(f[EVT].mean())
    b = (W @ fl) / (W @ nf)
    return dict(catch=float(fl.sum() / nf.sum()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                fa=[float(x) for x in fa])


def auroc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ruleB(S):
    m = S[LINE_T].mean(0); d = S[LINE_T].std(0)
    return ((S - m) / d).max(1)


def plain(S):
    m = np.array([ZM[f"tok{r}"]["mu"] for r in R]); d = np.array([ZM[f"tok{r}"]["sd"] for r in R])
    return ((S - m) / d).max(1)


def heads(S, kind, shuffle=False, seed=0):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = YF[TRAIN].copy()
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    sc = StandardScaler().fit(S[TRAIN])
    clf = LogisticRegression(C=1.0, max_iter=5000) if kind == "logit" else QuadraticDiscriminantAnalysis(reg_param=0.1)
    clf.fit(sc.transform(S[TRAIN]), y)
    return clf.decision_function(sc.transform(S))


def flag(score, idx=LINE_T, k=2):
    return score > line_of(score, idx, k)


def per_state(S, Sclean):
    out = {}
    rb = ruleB(S); pl = plain(S)
    out["ruleB"] = catch([flag(rb)]); out["ruleB_line"] = float(line_of(rb))
    out["ruleB_cal600"] = catch([flag(rb, CAL600, 6)])
    out["plain"] = catch([flag(pl)])
    clean_line = line_of(plain(Sclean))
    out["clean_rule"] = catch([pl > clean_line])
    for kind in ("logit", "qda"):
        h = heads(S, kind)
        out[kind] = catch([flag(h)])
        out[kind + "_auroc"] = auroc(h[EVAL], YF[EVAL])
    out["logit_shuf"] = [catch([flag(heads(S, "logit", True, s))])["catch"] for s in range(5)]
    own = [catch([flag(S[:, j])])["catch"] for j in range(20)]
    out["member_own_line"] = dict(zip([f"tok{r}" for r in R], own))
    out["ruleB_auroc"] = auroc(rb[EVAL], YF[EVAL])
    # twin pair rate: false above its true twin (pairs share city + pos on eval)
    out["member_auroc"] = [auroc(S[EVAL, j], YF[EVAL]) for j in range(20)]
    # spread diagnostics on the 200 line-set trues
    dA = S[LINE_T].std(0); dC = Sclean[LINE_T].std(0)
    zt = (S[LINE_T] - S[LINE_T].mean(0)) / dA
    out["d_ratio"] = (dA / dC).round(3).tolist()
    out["top2_z"] = np.sort(zt, 0)[::-1][:2].round(3).T.tolist()
    out["kurt"] = (((zt ** 4).mean(0)) - 3).round(3).tolist()
    # line null: line from 2000 resamples of the 200 line-set trues (Rule B moments refit per resample)
    rng = np.random.default_rng(126)
    li = np.where(LINE_T)[0]; c = []
    for _ in range(2000):
        bi = rng.choice(li, 200, replace=True)
        m = S[bi].mean(0); d = S[bi].std(0); z = ((S - m) / d).max(1)
        t = np.sort(z[bi])[::-1][1]
        c.append((z[EVF] > t).mean())
    out["ruleB_line_null"] = [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
    return out


if __name__ == "__main__":
    state = sys.argv[1]
    npz = sys.argv[sys.argv.index("--scores-npz") + 1] if "--scores-npz" in sys.argv else None
    clean_npz = sys.argv[sys.argv.index("--clean-npz") + 1] if "--clean-npz" in sys.argv else None
    S = scores_from_npz(npz) if npz else scores_from_ctok(state)
    Sc = scores_from_npz(clean_npz) if clean_npz else (scores_from_ctok("clean") if (D / "ctok/clean.npy").exists()
                                                       else scores_from_npz(D / "scores/clean.npz"))
    res = per_state(S, Sc)
    tag = os.environ.get("VERA_TAG", state)
    json.dump(res, open(HERE / f"vera26_{tag}.json", "w"), indent=1)
    print(state, json.dumps({k: res[k] for k in ("ruleB", "plain", "clean_rule", "logit", "qda", "logit_shuf",
                                                 "ruleB_auroc", "ruleB_line", "ruleB_line_null")}, indent=None))
