"""The frozen monitor panel: every monitor is fit on the CLEAN model only (fit pool = probe+attack+stop, 1400
statements, 350 cities) and then scores every model state with the same parameters.

Panel (names as in the post):
  tok<R>   country-token probe: StandardScaler + LogisticRegression(L2) on the last country-name token, readout R
           (R = 1..28 pre-norm block outputs, R = 29 = HF hidden_states[28] post-norm). C: 1020 d3.json choice for
           R <= 28; for R = 29 the same rule (5-fold GroupKFold-by-city OOF AUROC over {0.01, 0.1, 1, 10}, ties within
           0.002 -> smaller C).
  mean<R>  sentence-average probe: the same on the mean over tokens 1..n-1. C: 1020 hp.json for R <= 28; rule for 29.
  mlp14    small neural probe: the 1020 fit (data/ref/fits1020/final_mlp14.pt) on the standardised R = 14 mean.
  mega     megaprobe: the 1020 fit (final_mega.pt, k = 32) through the 1020 clean PCA (pca_clean.pt), readouts 1..28.
  base     the frozen 1019 probe (R = 14 mean), the target of last night's layer-14 fine-tune.
Stages:
  fit           -> data/fits/panel.npz (scaler mean/scale, coef, intercept for every tok<R>, mean<R>) + data/fits/C.json
  score <state> -> data/scores/<state>.npz (every panel monitor's score of all 2992 statements)
Run: .venv/bin/python panel.py fit ; .venv/bin/python panel.py score clean
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import DATA, REF, Acts, country_pos, device, load_split, pool_idx, rows, tokenize, MODEL

FITS = DATA / "fits"
SCORES = DATA / "scores"
READ = list(range(1, 30))
LIN_C = [0.01, 0.1, 1.0, 10.0]
TIE = 0.002
SEED = 1020
DEV = device()


def folds(idx, city, k):
    """1020 rule: GroupKFold by city, groups shuffled with seed 1020 first."""
    cs = sorted(set(city[idx]))
    perm = np.random.default_rng(SEED).permutation(len(cs))
    gid = {cs[j]: i for i, j in enumerate(perm)}
    g = np.array([gid[c] for c in city[idx]])
    return [(idx[a], idx[b]) for a, b in GroupKFold(k).split(idx, groups=g)]


def fit_lin(X, y, tr, C):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return sc, m


def select_lin(X, y, city, tr):
    res = {}
    for C in LIN_C:
        oof = np.zeros(len(y))
        for a, b in folds(tr, city, 5):
            sc, m = fit_lin(X, y, a, C)
            oof[b] = m.decision_function(sc.transform(X[b]))
        res[C] = float(roc_auc_score(y[tr], oof[tr]))
    best = max(res.values())
    return min(c for c in LIN_C if res[c] >= best - TIE), res


def tok_rows(A):
    """Flat memmap row of each statement's last country-name token."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenize(tk, [r["text"] for r in load_split()])
    pos, _ = country_pos(tk, ids)
    return A.off[:-1] + pos - 1  # token t lives at row off[i] + t - 1 (token 0 skipped)


def features(A, kind, R, trows):
    return A.at(R, trows) if kind == "tok" else A.mean(R)


def stage_fit():
    t0 = time.time()
    FITS.mkdir(parents=True, exist_ok=True)
    R_, y, city, split = rows()
    tr, _, _ = pool_idx()
    hp = json.loads((REF / "hp1020.json").read_text())
    d3 = json.loads((REF / "d3_1020.json").read_text())
    A = Acts("clean")
    trows = tok_rows(A)
    out, Cs = {}, {}
    for kind in ("tok", "mean"):
        for R in READ:
            X = features(A, kind, R, trows)
            if R <= 28:
                C = float((d3["C"][str(R)] if kind == "tok" else hp["lin"][str(R)])["C"])
                res = None
            else:
                C, res = select_lin(X, y, city, tr)
            sc, m = fit_lin(X, y, tr, C)
            nm = f"{kind}{R}"
            out[f"{nm}__mu"], out[f"{nm}__sd"] = sc.mean_, sc.scale_
            out[f"{nm}__w"], out[f"{nm}__b"] = m.coef_[0], np.array(m.intercept_[0])
            Cs[nm] = {"C": C, "cv_auroc": res}
            print(nm, C, f"{time.time() - t0:.0f}s", flush=True)
    out["tok_rows_clean"] = trows
    np.savez(FITS / "panel.npz", **out)
    (FITS / "C.json").write_text(json.dumps(Cs, indent=1))


def lin_apply(F, nm, X):
    return ((X - F[f"{nm}__mu"]) / F[f"{nm}__sd"]) @ F[f"{nm}__w"] + float(F[f"{nm}__b"])


# ---------------- megaprobe (1020 megaprobe-after probes.py, unchanged math) ----------------
def load_pca():
    pz = torch.load(REF / "fits1020" / "pca_clean.pt")
    return pz["mu"].to(DEV), pz["V"].to(DEV), pz["scale"].to(DEV)


def load_mega():
    return {a: t.to(DEV) for a, t in torch.load(REF / "fits1020" / "final_mega.pt")["P"].items()}


def mega_forward(P, X, M):
    """X (B, T, 28, k) whitened PCA coords, M (B, T) bool. Attention over all (token, layer) pairs."""
    a = torch.einsum("btlk,lk->btl", X, P["q"]) + P["c"]
    v = torch.einsum("btlk,lk->btl", X, P["w"])
    a = a.masked_fill(~M[:, :, None], -1e9).flatten(1).softmax(1)
    return (a * v.flatten(1)).sum(1) + P["b"], a.view_as(v)


def mega_scores(A, pca, P, mb=128):
    """Score every statement of a state: per statement its tokens 1..n-1 at readouts 1..28 -> whitened top-k."""
    mu, V, scale = pca
    k = P["w"].shape[1]
    lens = np.diff(A.off)
    sc, attl = np.zeros(A.n), np.zeros((A.n, 28))
    with torch.no_grad():
        for s in range(0, A.n, mb):
            e = min(A.n, s + mb)
            T = int(lens[s:e].max())
            X = torch.zeros(e - s, T, 28, k, device=DEV)
            M = torch.zeros(e - s, T, dtype=torch.bool, device=DEV)
            for j, i in enumerate(range(s, e)):
                H = torch.from_numpy(np.asarray(A.tok[A.off[i]:A.off[i + 1], 1:29], np.float32)).to(DEV)
                X[j, :lens[i]] = torch.einsum("cld,ldk->clk", H - mu, V[:, :, :k]) * scale[:, :k]
                M[j, :lens[i]] = True
            z, a = mega_forward(P, X, M)
            sc[s:e] = z.cpu().numpy()
            attl[s:e] = a.sum(1).cpu().numpy()
    return sc, attl


def mlp_net():
    c = torch.load(REF / "fits1020" / "final_mlp14.pt", weights_only=False)
    net = torch.nn.Sequential(torch.nn.Linear(1536, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
    net.load_state_dict(c["net"]); net.eval()
    return net, c["scaler_mean"], c["scaler_scale"]


def stage_score(state):
    t0 = time.time()
    SCORES.mkdir(parents=True, exist_ok=True)
    F = np.load(FITS / "panel.npz")
    A = Acts(state)
    trows = F["tok_rows_clean"]
    assert (A.off == Acts("clean").off).all()
    S = {}
    for R in READ:
        S[f"tok{R}"] = lin_apply(F, f"tok{R}", A.at(R, trows))
        Xm = A.mean(R)
        S[f"mean{R}"] = lin_apply(F, f"mean{R}", Xm)
        if R == 14:
            z = np.load(REF / "probe.npz")
            S["base"] = Xm.astype(np.float64) @ z["w"].astype(np.float64) + float(z["b"])
            net, m, sd = mlp_net()
            with torch.no_grad():
                S["mlp14"] = net(torch.from_numpy(((Xm - m) / sd).astype(np.float32)))[:, 0].numpy().astype(np.float64)
    S["mega"], S["mega_attn_layer"] = mega_scores(A, load_pca(), load_mega())
    np.savez(SCORES / f"{state}.npz", **S)
    print(state, "scored", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        stage_fit()
    else:
        stage_score(sys.argv[2])
