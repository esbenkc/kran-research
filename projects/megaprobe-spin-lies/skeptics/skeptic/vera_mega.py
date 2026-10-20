"""Vera's own refit of the verdict quantity, independent of build's code.
Arms (own implementations):
  a  : layer-12 mean over answer tokens, StandardScaler + LogisticRegression(C=0.05), class_weight balanced
  c' : attention probe over answer tokens AND the 24 layers on per-layer PCA-64 token features (vera_feats.py):
       4 heads; each head has softmax layer weights, a token query and a readout vector; logistic output.
       Hyperparameters FIXED a priori (no tuning): wd 1e-2, Adam lr 3e-3, batch 64, <=150 epochs, early stop on a
       20% inner split (patience 20, min 30 epochs), init scale 0.1, class-balanced loss.
Protocol: 5 folds grouped by prompt_id (my own map, seed FOLD_SEED, shared by all models), balanced n=400 draws from
the 4 training folds (all of a short class + fill), NDRAW draws, OOF score = mean over draws.
Optional SHUFFLE=1 permutes the training labels inside each draw (null).
    cd kran-research && FOLD_SEED=77 NDRAW=3 uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_mega.py base lam1_s0 ...
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, classes, load_labels  # noqa: E402
from vera_io import D20  # noqa: E402

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
FOLD_SEED = int(os.environ.get("FOLD_SEED", 77))
NDRAW = int(os.environ.get("NDRAW", 3))
SHUFFLE = int(os.environ.get("SHUFFLE", 0))
SUFFIX = os.environ.get("READ_SUFFIX", "_own")
TCAP = 128
LR = float(os.environ.get('LR', 1e-2))
WD = float(os.environ.get('WD', 1e-2))
MINEP = int(os.environ.get('MINEP', 60))
NROWS = 1000


def fold_map(pids, seed):
    u = sorted(set(pids)); rng = np.random.default_rng(seed); rng.shuffle(u)
    return {p: i % 5 for i, p in enumerate(u)}


def draw(tr, y, rng, n=400):
    pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
    hp = min(n // 2, len(pos)); hn = min(n // 2, len(neg))
    if hp < n // 2: hn = min(len(neg), n - hp)
    if hn < n // 2: hp = min(len(pos), n - hn)
    return np.concatenate([rng.choice(pos, hp, replace=False), rng.choice(neg, hn, replace=False)])


class Att(torch.nn.Module):
    def __init__(self, L, K, H=4):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(H, L))
        self.q = torch.nn.Parameter(torch.randn(H, K) * 0.1)
        self.v = torch.nn.Parameter(torch.randn(H, K) * 0.1)
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, m):  # x [B,T,L,K], m [B,T] bool
        a = torch.softmax(self.beta, -1)                      # [H,L]
        z = torch.einsum("btlk,hl->bhtk", x, a)               # [B,H,T,K]
        s = torch.einsum("bhtk,hk->bht", z, self.q)
        s = s.masked_fill(~m[:, None, :], -1e9)
        w = torch.softmax(s, -1)
        p = torch.einsum("bht,bhtk->bhk", w, z)
        return (p * self.v[None]).sum((1, 2)) + self.b


def fit_att(X, M, y, idx, rng, seed):
    torch.manual_seed(seed)
    perm = rng.permutation(idx); nv = len(perm) // 5
    va, tr = perm[:nv], perm[nv:]
    # standardise each (layer, component) on training tokens
    s1 = 0.0; s2 = 0.0; cnt = 0
    for j in range(0, len(tr), 32):  # chunked token stats, keeps RAM low
        xt = X[tr[j:j + 32]].astype(np.float32)[M[tr[j:j + 32]]]
        s1 = s1 + xt.sum(0); s2 = s2 + (xt.astype(np.float64) ** 2).sum(0); cnt += len(xt)
    mu = (s1 / cnt).astype(np.float32); sd = (np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)) + 1e-4).astype(np.float32)
    def batch(ii):
        t = int(M[ii].sum(1).max())  # trim padding to the longest answer in the batch
        m = M[ii, :t]
        return torch.from_numpy((X[ii, :t].astype(np.float32) - mu) / sd) * torch.from_numpy(m)[..., None, None], torch.from_numpy(m)
    L, K = X.shape[2], X.shape[3]
    net = Att(L, K); opt = torch.optim.Adam([{'params': [net.beta, net.b], 'weight_decay': 0.0},
                            {'params': [net.q, net.v], 'weight_decay': WD}], lr=LR)
    yt = torch.from_numpy(y.astype(np.float32))
    wpos = (len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(wpos)))
    xv, mv = batch(va); best = (1e9, None); bad = 0
    for ep in range(150):
        net.train()
        for s in range(0, len(tr), 64):
            ii = tr[rng.permutation(len(tr))[:64]] if False else tr[s:s + 64]
            xb, mb = batch(ii)
            loss = lossf(net(xb, mb), yt[ii]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xv, mv), yt[va]).item()
        if vl < best[0] - 1e-4:
            best = (vl, {k: t.clone() for k, t in net.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= 20 and ep >= MINEP: break
        rng.shuffle(tr)
    net.load_state_dict(best[1]); net.eval()
    def score(ii):
        out = []
        with torch.no_grad():
            for s in range(0, len(ii), 128):
                xb, mb = batch(ii[s:s + 128]); out.append(net(xb, mb).numpy())
        return np.concatenate(out)
    return score


def load_read(read):
    f = np.load(HERE / "feats" / f"{read}.npz")
    off = f["off"][:NROWS + 1]; pca = f["pca"][:off[-1]]
    X = np.zeros((NROWS, TCAP, pca.shape[1], pca.shape[2]), np.float16); M = np.zeros((NROWS, TCAP), bool)
    for i in range(NROWS):
        t = pca[off[i]:off[i + 1]][:TCAP]; X[i, :len(t)] = t; M[i, :len(t)] = True
    del pca
    return f["meanL"][:NROWS, 11], X, M  # layer 12 (hidden_states[12] = index 11) for arm a


def run(model, fm):
    L = load_labels(model, D20)[:NROWS]; c = classes(L)
    y = c["lie"].astype(int); keep = np.array([r["y_judge"] is not None for r in L])
    folds = np.array([fm[r["prompt_id"]] for r in L])
    read = model + SUFFIX if not model.endswith(SUFFIX) else model
    m12, X, M = load_read(read)
    sa = np.zeros(NROWS); sc = np.zeros(NROWS); t0 = time.time()
    for f in range(5):
        tr_all = np.flatnonzero((folds != f) & keep); te = np.flatnonzero(folds == f)
        for d in range(NDRAW):
            rng = np.random.default_rng(1000 * FOLD_SEED + 10 * f + d)
            idx = draw(tr_all, y, rng); yy = y.copy()
            if SHUFFLE: yy[idx] = rng.permutation(yy[idx])
            sc_ = StandardScaler().fit(m12[idx])
            lr = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc_.transform(m12[idx]), yy[idx])
            sa[te] += lr.decision_function(sc_.transform(m12[te])) / NDRAW
            score = fit_att(X, M, yy, idx, rng, seed=FOLD_SEED * 100 + f * 10 + d)
            sc[te] += score(te) / NDRAW
    r = dict(model=model, secs=round(time.time() - t0),
             a_spin_fs=auroc(sa[c["spin"]], sa[c["truthful_fs"]]), c_spin_fs=auroc(sc[c["spin"]], sc[c["truthful_fs"]]),
             a_spin_all=auroc(sa[c["spin"]], sa[c["truthful"]]), c_spin_all=auroc(sc[c["spin"]], sc[c["truthful"]]),
             a_lies=auroc(sa[c["lie"]], sa[c["truthful"]]), c_lies=auroc(sc[c["lie"]], sc[c["truthful"]]))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return r, sa, sc


if __name__ == "__main__":
    models = sys.argv[1:]
    pids = set()
    for m in models:
        pids |= {r["prompt_id"] for r in load_labels(m, D20)[:NROWS]}
    fm = fold_map(pids, FOLD_SEED)
    tag = f"fs{FOLD_SEED}_d{NDRAW}_sh{SHUFFLE}{SUFFIX}"
    out = {}
    for m in models:
        r, sa, sc = run(m, fm)
        out[m] = dict(r, sa=sa.tolist(), sc=sc.tolist())
        json.dump(out, open(HERE / f"vera_mega_{tag}.json", "w"))
