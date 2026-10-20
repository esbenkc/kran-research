"""Monitors of PREREG.md s3-s4: fit on the CLEAN state's training pool only, then score every state.

Stages (run in this order; the select stage reads only clean data):
  uv run python probes.py pool      # mean-pooled features per state -> data/pooled/<state>.npy (2992, 29, 1536) fp32
  uv run python probes.py select    # s4 grids on clean: lin_l C, mlp14 weight decay, mega k x L2 -> data/hp.json,
                                    # convergence + label-count fits (clean) -> data/fits/
  uv run python probes.py score     # final fits on the 1400 clean pool; score clean + s0..s2 -> data/scores.npz

Labels: y = 1 for a FALSE statement (the class the monitor flags).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import ACTS, ATTACK_LAYER, DATA, STATES, Acts, device, load_probe, load_split

SEED = 1020
LAYERS = list(range(1, 29))  # hidden_states[1..28] = the 28 block outputs
POOLED = DATA / "pooled"
FITS = DATA / "fits"
HP_FILE = DATA / "hp.json"
DEV = device()
LIN_C = [0.01, 0.1, 1.0, 10.0]
MLP_WD = [1e-4, 1e-3, 1e-2]
MEGA_K = [32, 128]
MEGA_L2 = [1e-3, 1e-2, 3e-2]
CURVE_N = [200, 400, 800, 1400]
TIE = 0.002


# ---------------- rows ----------------
def rows():
    R = load_split()
    y = np.array([1 - r["true"] for r in R])
    city = np.array([r["city"] for r in R])
    split = np.array([r["split"] for r in R])
    return R, y, city, split


def pool_idx():
    R, y, city, split = rows()
    tr = np.where(np.isin(split, ["probe", "attack", "stop"]))[0]
    line = np.where((split == "refit") & (y == 0))[0]
    ev = np.where(split == "eval")[0]
    return tr, line, ev


def curve_subsets():
    """Nested city subsets of the training pool: first n/4 cities of a seed-1020 permutation (each city = 2+2)."""
    R, y, city, split = rows()
    tr, _, _ = pool_idx()
    cities = sorted(set(city[tr]))
    perm = np.random.default_rng(SEED).permutation(len(cities))
    order = [cities[i] for i in perm]
    out = {}
    for n in CURVE_N:
        keep = set(order[:n // 4])
        out[n] = np.array([i for i in tr if city[i] in keep])
        assert len(out[n]) == n and y[out[n]].sum() == n // 2
    return out


def inner_split(idx, city, frac=0.2, seed=SEED):
    """City-grouped inner split: ~frac of the cities of idx go to validation."""
    cs = sorted(set(city[idx]))
    perm = np.random.default_rng(seed).permutation(len(cs))
    val_c = {cs[i] for i in perm[:max(1, int(round(frac * len(cs))))]}
    v = np.array([c in val_c for c in city[idx]])
    return idx[~v], idx[v]


def folds(idx, city, k):
    """GroupKFold by city; groups shuffled with seed 1020 first (GroupKFold itself is deterministic)."""
    cs = sorted(set(city[idx]))
    perm = np.random.default_rng(SEED).permutation(len(cs))
    gid = {cs[j]: i for i, j in enumerate(perm)}
    g = np.array([gid[c] for c in city[idx]])
    return [(idx[a], idx[b]) for a, b in GroupKFold(k).split(idx, groups=g)]


# ---------------- pooled features ----------------
def pool_all():
    POOLED.mkdir(parents=True, exist_ok=True)
    for st in STATES:
        f = POOLED / f"{st}.npy"
        if f.exists():
            continue
        A = Acts(st)
        X = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(A.n, A.L, A.tok.shape[2]))
        for l in range(A.L):
            X[:, l] = A.pooled(l, "mean")
        X.flush()
        print(st, "pooled", flush=True)


def pooled(st, l):
    return np.asarray(np.load(POOLED / f"{st}.npy", mmap_mode="r")[:, l], np.float32)


# ---------------- lin_l ----------------
def fit_lin(X, y, tr, C):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return sc, m


def lin_score(sc, m, X):
    return m.decision_function(sc.transform(X))


def select_lin(X, y, city, tr):
    res = {}
    F = folds(tr, city, 5)
    for C in LIN_C:
        oof = np.zeros(len(y))
        for a, b in F:
            sc, m = fit_lin(X, y, a, C)
            oof[b] = lin_score(sc, m, X[b])
        res[C] = float(roc_auc_score(y[tr], oof[tr]))
    best = max(res.values())
    C = min(c for c in LIN_C if res[c] >= best - TIE)  # ties -> smaller C
    return C, res


# ---------------- mlp14 ----------------
def fit_mlp(X, y, city, tr, wd, seed=SEED, epochs=200, patience=20, bs=64):
    """X already standardised. Early stop on a 20% city-grouped inner split (BCE), Adam lr 1e-3, batch 64."""
    torch.manual_seed(seed)
    a, v = inner_split(tr, city, seed=seed)
    d = X.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    Xa, ya = torch.from_numpy(X[a]), torch.tensor(y[a], dtype=torch.float32)
    Xv, yv = torch.from_numpy(X[v]), torch.tensor(y[v], dtype=torch.float32)
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(seed)
    best, state, bad, log = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(len(a)))
        for s in range(0, len(a), bs):
            b = order[s:s + bs]
            loss = lossf(net(Xa[b])[:, 0], ya[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv)[:, 0], yv))
            tl = float(lossf(net(Xa)[:, 0], ya))
        log.append((ep, tl, vl))
        if vl < best - 1e-6:
            best, state, bad = vl, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, log


def mlp_score(net, X):
    with torch.no_grad():
        return net(torch.from_numpy(np.ascontiguousarray(X, np.float32)))[:, 0].numpy().astype(np.float64)


# ---------------- mega (megaprobe-spin-lies fit_c, adapted to statements) ----------------
def pca_fit(acts: Acts, tok_rows, kmax=128, chunk=2048):
    """Per layer (1..28) PCA of the clean training-pool tokens (no labels). -> mu (L,d), V (L,d,kmax), scale (L,kmax)."""
    L, d = len(LAYERS), acts.tok.shape[2]
    S = torch.zeros(L, d, dtype=torch.float64)
    C = torch.zeros(L, d, d, dtype=torch.float32, device=DEV)
    n, ref = 0, None
    for s in range(0, len(tok_rows), chunk):
        r = tok_rows[s:s + chunk]
        X = torch.from_numpy(np.asarray(acts.tok[r][:, LAYERS], np.float32)).to(DEV)  # (c, L, d)
        if ref is None:
            ref = X.mean(0)
        X = X - ref
        S += X.sum(0).cpu().double()
        C += torch.einsum("cld,cle->lde", X, X)
        n += X.shape[0]
    m_s = S / n
    cov = C.cpu().double() / n - torch.einsum("ld,le->lde", m_s, m_s)
    del C
    ev, V = torch.linalg.eigh(cov)  # ascending
    mu = m_s.float().to(DEV) + ref
    if DEV.type == "mps":
        torch.mps.empty_cache()
    return mu, V[:, :, -kmax:].float().flip(-1).to(DEV), ev[:, -kmax:].flip(-1).float().clamp_min(1e-6).rsqrt().to(DEV)


def pca_project(acts: Acts, pca, k, chunk=2048):
    """Whitened top-k projection of every token of every statement, laid out padded per statement:
    -> (Zp (n_stmt, T_max, L, k) fp16 on DEV, M (n_stmt, T_max) bool on DEV). Padding positions are masked out."""
    mu, V, scale = pca
    Z = torch.empty(acts.tok.shape[0], len(LAYERS), k, dtype=torch.float16, device=DEV)
    for s in range(0, acts.tok.shape[0], chunk):
        X = torch.from_numpy(np.asarray(acts.tok[s:s + chunk][:, LAYERS], np.float32)).to(DEV)
        Z[s:s + chunk] = (torch.einsum("cld,ldk->clk", X - mu, V[:, :, :k]) * scale[:, :k]).half()
    lens = np.diff(acts.off)
    T = int(lens.max())
    pos = np.concatenate([np.arange(n) for n in lens])
    stm = np.repeat(np.arange(acts.n), lens)
    Zp = torch.zeros(acts.n, T, len(LAYERS), k, dtype=torch.float16, device=DEV)
    Zp[torch.from_numpy(stm).to(DEV), torch.from_numpy(pos).to(DEV)] = Z
    M = torch.from_numpy(np.arange(T)[None, :] < lens[:, None]).to(DEV)
    del Z
    if DEV.type == "mps":
        torch.mps.empty_cache()
    return Zp, M


def batch(ZM, idx, k):
    Zp, M = ZM
    ii = torch.from_numpy(np.asarray(idx)).to(DEV)
    Mb = M[ii]
    T = int(Mb.sum(1).max())
    return Zp[ii, :T, :, :k].float(), Mb[:, :T]


def mega_forward(P, X, M, attn=False):
    a = torch.einsum("btlk,lk->btl", X, P["q"]) + P["c"]
    v = torch.einsum("btlk,lk->btl", X, P["w"])
    a = a.masked_fill(~M[:, :, None], -1e9).flatten(1).softmax(1)
    s = (a * v.flatten(1)).sum(1) + P["b"]
    return (s, a.view_as(v)) if attn else s


def mega_score(P, ZM, idx, mb=400, attn=False):
    """Scores of statements idx; attn=True also returns each statement's (T_i, L) attention weights."""
    k = P["w"].shape[1]
    out, att = [], []
    with torch.no_grad():
        for s in range(0, len(idx), mb):
            X, M = batch(ZM, idx[s:s + mb], k)
            r = mega_forward(P, X, M, attn=attn)
            if attn:
                out.append(r[0].cpu().numpy())
                att += [x[:int(m.sum())].cpu().numpy() for x, m in zip(r[1], M)]
            else:
                out.append(r.cpu().numpy())
    sc = np.concatenate(out).astype(np.float64)
    return (sc, att) if attn else sc


def fit_mega(ZM, y, city, tr, k, l2, seed=SEED, epochs=300, lr=1e-2, mb=400):
    """s3: Adam lr 1e-2, L2 on q and w, class-balanced BCE, 400-row minibatches, 300 epochs; checkpoint every 10
    epochs on a 20% city-grouped inner split: best inner AUROC, ties -> lower inner BCE, then the later epoch."""
    a, v = inner_split(tr, city, seed=seed)
    rng = np.random.default_rng(seed)
    L = ZM[0].shape[2]
    g = torch.Generator().manual_seed(seed)
    P = {"q": (torch.randn(L, k, generator=g) * 0.01).to(DEV).requires_grad_(),
         "c": torch.zeros(L, device=DEV, requires_grad=True),
         "w": (torch.randn(L, k, generator=g) * 0.01).to(DEV).requires_grad_(),
         "b": torch.zeros((), device=DEV, requires_grad=True)}
    opt = torch.optim.Adam(P.values(), lr=lr)
    ya = torch.tensor(y[a], dtype=torch.float32, device=DEV)
    pw = (ya == 0).sum() / (ya == 1).sum().clamp_min(1)
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    best, log = None, []
    for ep in range(epochs):
        order = rng.permutation(len(a))
        for s in range(0, len(a), mb):
            bi = np.sort(order[s:s + mb])
            X, M = batch(ZM, a[bi], k)
            z = mega_forward(P, X, M)
            loss = lossf(z, ya[torch.from_numpy(bi).to(DEV)], pos_weight=pw) \
                + l2 * (P["q"].pow(2).sum() + P["w"].pow(2).sum())
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            sv, sa = mega_score(P, ZM, v), mega_score(P, ZM, a)
            au = float(roc_auc_score(y[v], sv))
            vb = float(lossf(torch.from_numpy(sv), torch.tensor(y[v], dtype=torch.float64)))
            tb = float(lossf(torch.from_numpy(sa), torch.tensor(y[a], dtype=torch.float64)))
            log.append({"epoch": ep + 1, "train_bce": tb, "inner_bce": vb, "inner_auroc": au})
            key = (au, -vb, ep)
            if best is None or key >= best[0]:
                best = (key, {n: t.detach().clone() for n, t in P.items()}, ep + 1)
    if DEV.type == "mps":
        torch.mps.empty_cache()
    return best[1], best[2], log


# ---------------- stages ----------------
def stage_select():
    t0 = time.time()
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    FITS.mkdir(parents=True, exist_ok=True)
    hp = {"rules": "PREREG s4", "lin": {}, "mlp14": {}, "mega": {}}
    # lin_l: C per layer (5-fold GroupKFold by city, OOF AUROC; ties within 0.002 -> smaller C)
    for l in LAYERS:
        X = pooled("clean", l)
        C, res = select_lin(X, y, city, tr)
        hp["lin"][l] = {"C": C, "cv_auroc": res}
        print("lin", l, C, res, flush=True)
    # mlp14
    X = pooled("clean", ATTACK_LAYER)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X).astype(np.float32)
    res = {}
    for wd in MLP_WD:
        oof = np.zeros(len(y))
        for a, b in folds(tr, city, 4):
            net, _ = fit_mlp(Xs, y, city, a, wd)
            oof[b] = mlp_score(net, Xs[b])
        res[wd] = float(roc_auc_score(y[tr], oof[tr]))
        print("mlp14", wd, res[wd], flush=True)
    best = max(res.values())
    hp["mlp14"] = {"wd": max(w for w in MLP_WD if res[w] >= best - TIE), "cv_auroc": res}
    # mega
    A = Acts("clean")
    tok_rows = np.concatenate([np.arange(A.off[i], A.off[i + 1]) for i in tr])
    pca = pca_fit(A, tok_rows)
    torch.save({"mu": pca[0].cpu(), "V": pca[1].cpu(), "scale": pca[2].cpu()}, FITS / "pca_clean.pt")
    ZM = pca_project(A, pca, max(MEGA_K))
    grid = []

    def run_cell(k, l2):
        oof = np.zeros(len(y)); secs = []
        for j, (a, b) in enumerate(folds(tr, city, 4)):
            t1 = time.time()
            P, ep, _ = fit_mega(ZM, y, city, a, k, l2, seed=SEED + j)
            oof[b] = mega_score(P, ZM, b)
            secs.append(time.time() - t1)
        cell = {"k": k, "l2": l2, "cv_auroc": float(roc_auc_score(y[tr], oof[tr])), "sec_per_fit": float(np.mean(secs))}
        grid.append(cell)
        print("mega", cell, flush=True)

    for k in MEGA_K:
        for l2 in MEGA_L2:
            run_cell(k, l2)

    def pick():
        b = max(c["cv_auroc"] for c in grid)
        cand = [c for c in grid if c["cv_auroc"] >= b - TIE]
        return sorted(cand, key=lambda c: (c["k"], -c["l2"]))[0]  # ties -> smaller k, then larger L2

    ch = pick()
    l2s = sorted({c["l2"] for c in grid if c["k"] == ch["k"]})
    if ch["l2"] in (l2s[0], l2s[-1]):  # one edge extension past the winning edge
        run_cell(ch["k"], ch["l2"] / 3 if ch["l2"] == l2s[0] else ch["l2"] * 3)
        hp["mega_extended"] = True
        ch = pick()
    hp["mega"] = {"k": ch["k"], "l2": ch["l2"], "grid": grid}
    HP_FILE.write_text(json.dumps(hp, indent=1))
    print("hp written", {"mlp14": hp["mlp14"]["wd"], "mega": (ch["k"], ch["l2"])}, f"{time.time() - t0:.0f}s", flush=True)


def stage_curves():
    """Convergence + label-count fits on CLEAN (s9): lin_14, mlp14, mega at the chosen HP on nested subsets.
    Saves the fitted parameters so the score stage can score them on attacked states."""
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    hp = json.loads(HP_FILE.read_text())
    subs = curve_subsets()
    A = Acts("clean")
    pz = torch.load(FITS / "pca_clean.pt")
    pca = (pz["mu"].to(DEV), pz["V"].to(DEV), pz["scale"].to(DEV))
    k, l2 = hp["mega"]["k"], hp["mega"]["l2"]
    ZM = pca_project(A, pca, max(MEGA_K))
    X14 = pooled("clean", ATTACK_LAYER)
    out = {"lin14": {}, "mlp14": {}, "mega": {}}
    for n, idx in subs.items():
        sc, m = fit_lin(X14, y, idx, hp["lin"][str(ATTACK_LAYER)]["C"])
        torch.save({"scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "coef": m.coef_[0], "intercept": m.intercept_[0]},
                   FITS / f"curve_lin14_n{n}.pt")
        out["lin14"][n] = {"clean_eval_auroc": float(roc_auc_score(y[ev], lin_score(sc, m, X14[ev])))}
        scm = StandardScaler().fit(X14[idx]); Xs = scm.transform(X14).astype(np.float32)
        net, log = fit_mlp(Xs, y, city, idx, hp["mlp14"]["wd"])
        torch.save({"net": net.state_dict(), "scaler_mean": scm.mean_, "scaler_scale": scm.scale_, "log": log},
                   FITS / f"curve_mlp14_n{n}.pt")
        out["mlp14"][n] = {"clean_eval_auroc": float(roc_auc_score(y[ev], mlp_score(net, Xs[ev]))), "epochs": len(log)}
        P, ep, log = fit_mega(ZM, y, city, idx, k, l2)
        torch.save({"P": {a: t.cpu() for a, t in P.items()}, "epoch": ep, "log": log}, FITS / f"curve_mega_n{n}.pt")
        out["mega"][n] = {"clean_eval_auroc": float(roc_auc_score(y[ev], mega_score(P, ZM, ev))), "best_epoch": ep}
        print(n, {a: out[a][n] for a in out}, flush=True)
    (FITS / "curves_clean.json").write_text(json.dumps(out, indent=1))


def load_pca():
    pz = torch.load(FITS / "pca_clean.pt")
    return pz["mu"].to(DEV), pz["V"].to(DEV), pz["scale"].to(DEV)


def stage_score():
    """Final fits on the 1400-statement clean pool at the hp.json settings, then score EVERY state (clean, s0-s2) on
    all 2992 statements with the SAME fitted monitors (and the same clean PCA for mega). -> data/scores.npz"""
    t0 = time.time()
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    hp = json.loads(HP_FILE.read_text())
    S = {"y": y, "split": split, "city": city, "src": np.array([r["src"] for r in R]),
         "pos": np.array([r["pos"] for r in R])}
    # base: the frozen 1019 probe at hidden_states[14], mean pool
    w, b = load_probe()
    for st in STATES:
        S[f"{st}__base"] = pooled(st, ATTACK_LAYER).astype(np.float64) @ w + b
    # lin_l
    for l in LAYERS:
        Xc = pooled("clean", l)
        sc, m = fit_lin(Xc, y, tr, hp["lin"][str(l)]["C"])
        for st in STATES:
            S[f"{st}__lin{l}"] = lin_score(sc, m, Xc if st == "clean" else pooled(st, l))
    print("lin done", f"{time.time() - t0:.0f}s", flush=True)
    # mlp14
    Xc = pooled("clean", ATTACK_LAYER)
    scm = StandardScaler().fit(Xc[tr])
    net, log = fit_mlp(scm.transform(Xc).astype(np.float32), y, city, tr, hp["mlp14"]["wd"])
    torch.save({"net": net.state_dict(), "scaler_mean": scm.mean_, "scaler_scale": scm.scale_, "log": log},
               FITS / "final_mlp14.pt")
    curves = {n: torch.load(FITS / f"curve_mlp14_n{n}.pt", weights_only=False) for n in CURVE_N}
    lcurves = {n: torch.load(FITS / f"curve_lin14_n{n}.pt", weights_only=False) for n in CURVE_N}
    for st in STATES:
        X = pooled(st, ATTACK_LAYER)
        S[f"{st}__mlp14"] = mlp_score(net, scm.transform(X).astype(np.float32))
        for n in CURVE_N:
            c = curves[n]
            cn = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
            cn.load_state_dict(c["net"]); cn.eval()
            S[f"{st}__mlp14_n{n}"] = mlp_score(cn, ((X - c["scaler_mean"]) / c["scaler_scale"]).astype(np.float32))
            lc = lcurves[n]
            S[f"{st}__lin14_n{n}"] = ((X - lc["scaler_mean"]) / lc["scaler_scale"]) @ lc["coef"] + lc["intercept"]
    # shuffled-label nulls for lin_14 (5 permutations of the training labels, seeds 1..5)
    Xc = pooled("clean", ATTACK_LAYER)
    for p in range(1, 6):
        yp = y.copy(); yp[tr] = y[tr][np.random.default_rng(p).permutation(len(tr))]
        sc, m = fit_lin(Xc, yp, tr, hp["lin"][str(ATTACK_LAYER)]["C"])
        for st in STATES:
            S[f"{st}__lin14_null{p}"] = lin_score(sc, m, Xc if st == "clean" else pooled(st, ATTACK_LAYER))
    print("mlp/lin14 curves + nulls done", f"{time.time() - t0:.0f}s", flush=True)
    # mega: final fit, curve fits, nulls; all scored through the clean PCA
    pca = load_pca()
    k, l2 = hp["mega"]["k"], hp["mega"]["l2"]
    ZMc = pca_project(Acts("clean"), pca, max(MEGA_K))
    P, ep, log = fit_mega(ZMc, y, city, tr, k, l2)
    torch.save({"P": {a: t.cpu() for a, t in P.items()}, "epoch": ep, "log": log}, FITS / "final_mega.pt")
    nulls = {}
    for p in range(1, 6):
        yp = y.copy(); yp[tr] = y[tr][np.random.default_rng(p).permutation(len(tr))]
        nulls[p], _, _ = fit_mega(ZMc, yp, city, tr, k, l2)
    del ZMc
    cps = {n: {a: t.to(DEV) for a, t in torch.load(FITS / f"curve_mega_n{n}.pt")["P"].items()} for n in CURVE_N}
    allidx = np.arange(len(y))
    att_layer, att_tok = {}, {}
    for st in STATES:
        A = Acts(st)
        ZM = pca_project(A, pca, max(MEGA_K))
        s, att = mega_score(P, ZM, allidx, attn=True)
        S[f"{st}__mega"] = s
        att_layer[st] = np.stack([a.sum(0) for a in att])  # (n_stmt, 28): attention mass per layer
        S[f"{st}__mega_attn_layer"] = att_layer[st]
        S[f"{st}__mega_attn_tok"] = np.concatenate([a.sum(1) for a in att])  # (n_tok,): mass per token
        for n in CURVE_N:
            S[f"{st}__mega_n{n}"] = mega_score(cps[n], ZM, allidx)
        for p in range(1, 6):
            S[f"{st}__mega_null{p}"] = mega_score(nulls[p], ZM, allidx)
        del ZM
        if DEV.type == "mps":
            torch.mps.empty_cache()
        print(st, "mega scored", f"{time.time() - t0:.0f}s", flush=True)
    S["tok_off"] = Acts("clean").off
    np.savez(DATA / "scores.npz", **S)
    print("scores written", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"pool": pool_all, "select": stage_select, "curves": stage_curves, "score": stage_score}[sys.argv[1]]()
