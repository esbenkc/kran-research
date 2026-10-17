"""Vera's independent loader + predictor (reads only build's saved data files, never its code)."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, norm

_HERE = Path(__file__).resolve()
# works both from swarm-<room>/skeptic/ and from kran-research/projects/<slug>/skeptics/skeptic/
D = next(p for p in [_HERE.parents[2] / "data", _HERE.parents[2] / "kran-research/projects/fisher-pullback-transfer/data"]
         if (p / "selection.json").exists())

def auroc(s, y):
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load():
    probes = json.load(open(D / "probes/probes.json"))
    sel = json.load(open(D / "selection.json"))
    W = np.load(D / "probes/probe_weights.npz")["W"]
    name2i = {p["name"]: p["idx"] for p in probes}
    acts = {}
    for p in probes:
        src = p["source"]
        if src not in acts:
            z = np.load(D / f"clean/acts_{src}.npz")
            acts[src] = dict(X=z["X"].astype(np.float64), label=z["label"], split=z["split"])
    fz = np.load(D / "fisher/fisher_sigma.npz")
    return dict(probes=probes, sel=sel, W=W, name2i=name2i, acts=acts, fz=fz)


def test_set(L, name, test_code=1):
    p = L["probes"][L["name2i"][name]]
    a = L["acts"][p["source"]]
    m = a["split"] == test_code
    return a["X"][m], (a["label"][m] == p["label"])


def metrics(L, c=1.0, F=None):
    fz = L["fz"]
    F = fz["F_common"] if F is None else F
    ev = np.linalg.eigvalsh(F)
    lam = c * np.median(ev)
    PF = np.linalg.inv(F + lam * np.eye(F.shape[0]))
    from sklearn.covariance import LedoitWolf
    S = LedoitWolf().fit(fz["pooled_retain"].astype(np.float64)).covariance_
    return {"plain": np.eye(F.shape[0]), "whitened": S, "fisher": PF}, lam


def r_P(P, u, w):
    return (u @ P @ w) / (w @ P @ w)


def pred_drop_exact(Xu, yu, u, w, ratio, r):
    """ratio = delta_w / G_w(w). Predicted attacked score s_u + ratio * r * s_w on u's clean test set."""
    su, sw = Xu @ u, Xu @ w
    return auroc(su, yu) - auroc(su + ratio * r * sw, yu)


def pred_drop_gauss(Xu, yu, u, w, ratio, r):
    su, sw = Xu @ u, Xu @ w
    gap = su[yu].mean() - su[~yu].mean()
    n1, n0 = yu.sum(), (~yu).sum()
    sd = np.sqrt(((n1 - 1) * su[yu].var(ddof=1) + (n0 - 1) * su[~yu].var(ddof=1)) / (n1 + n0 - 2))
    Gu = sw[yu].mean() - sw[~yu].mean()
    new = gap + ratio * r * Gu
    return norm.cdf(gap / sd / np.sqrt(2)) - norm.cdf(new / sd / np.sqrt(2))


def r2(pred, meas):
    pred, meas = np.asarray(pred), np.asarray(meas)
    return 1 - ((meas - pred) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
