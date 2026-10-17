"""T16 step 0 (Iris): ceiling of 'candidate score = linear functional of x, per (rank, position) of candidate'.

Candidates = top-K of the phantom-augmented order (phantom 0 at position -1 -> output 0).
score_k = A[k, pos_k] . F(x), F = feature bank shared by all candidates. Fitted (discovery only).
Train dev seed 81, test dev seed 82. Target = model argmax (marginal over candidates with that output).
Usage: .venv/bin/python residual_ceiling.py --K 3 --feats x,relu,sorted
"""
import argparse, json, time
import numpy as np, torch
from residual_common import data, c1, aug_order

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=3)
p.add_argument("--feats", default="x,relu")
p.add_argument("--ntr", type=int, default=2_000_000)
p.add_argument("--nte", type=int, default=500_000)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--posdep", default="self", help="self | self+max | cfg (A indexed by candidate rank and whole (pM,pS,pT))")
a = p.parse_args()
torch.manual_seed(0)


def feats(X):
    fs = []
    sc = np.sqrt((X ** 2).mean(1, keepdims=True))  # scale, to give rank offsets homogeneous units
    for f in a.feats.split(","):
        if f == "x": fs.append(X)
        elif f == "relu": fs.append(np.maximum(X, 0))
        elif f == "neg": fs.append(np.minimum(X, 0))
        elif f == "sorted": fs.append(-np.sort(-np.concatenate([np.zeros((len(X), 1)), X], 1), 1))
        elif f == "scale": fs.append(sc)
        else: raise ValueError(f)
    return np.concatenate(fs, 1)


def prep(seed, n):
    X, L, m = data(seed, n)
    o = aug_order(X)[:, :a.K]  # (n,K) positions -1..9
    F = feats(X)
    outpos = np.maximum(o, 0)
    tgt = outpos == m[:, None]  # (n,K) bool; multiple true only for phantom/x0 ambiguity
    idx = o + 1  # 0..10
    if a.posdep == "self+max":
        pmx = o[:, :1] + 1
        idx = idx * 11 + pmx
    elif a.posdep == "cfg":  # whole (pM,pS,pT) config, same index for every candidate slot
        c = aug_order(X)[:, :3] + 1
        idx = np.repeat((c[:, 0] * 121 + c[:, 1] * 11 + c[:, 2])[:, None], a.K, 1)
    return X, m, torch.tensor(F, dtype=torch.float32), torch.tensor(idx), torch.tensor(tgt), outpos


t0 = time.time()
Xtr, mtr, Ftr, Itr, Ttr, Otr = prep(81, a.ntr)
Xte, mte, Fte, Ite, Tte, Ote = prep(82, a.nte)
d = Ftr.shape[1]
npos = {"self": 11, "self+max": 121, "cfg": 1331}[a.posdep]
A = torch.zeros(a.K, npos, d, requires_grad=True)
with torch.no_grad():  # init: score = own value (x feature) -> exact top-K order; rank-1 candidate wins via offset
    pass
off = torch.zeros(a.K, requires_grad=True)
opt = torch.optim.Adam([A, off], lr=3e-3)
kk = torch.arange(a.K)


def scores(F, I):
    W = A[kk[None, :], I]  # (n,K,d)
    return torch.einsum("nkd,nd->nk", W, F) + off


def evaluate(F, I, T, m, O):
    with torch.no_grad():
        s = torch.cat([scores(F[i:i + 200000], I[i:i + 200000]) for i in range(0, len(F), 200000)])
    pick = O[np.arange(len(O)), s.argmax(1).numpy()]
    return float((pick == m).mean()), pick


bs = 8192
for ep in range(a.epochs):
    perm = torch.randperm(len(Ftr))
    for i in range(0, len(perm), bs):
        b = perm[i:i + bs]
        s = scores(Ftr[b], Itr[b])
        lse = torch.logsumexp(s, 1)
        tl = torch.logsumexp(s.masked_fill(~Ttr[b], -1e9), 1)
        valid = Ttr[b].any(1)
        loss = (lse - tl)[valid].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for g in opt.param_groups: g["lr"] *= 0.6
    tr, _ = evaluate(Ftr[:500000], Itr[:500000], Ttr[:500000], mtr[:500000], Otr[:500000])
    te, pk = evaluate(Fte, Ite, Tte, mte, Ote)
    print(f"ep {ep} loss {loss.item():.4f} train {tr:.5f} test {te:.5f} ({time.time()-t0:.0f}s)", flush=True)

cov = float(Tte.any(1).float().mean())
err = mte != np.argsort(Xte, 1)[:, -2]
res = dict(K=a.K, feats=a.feats, posdep=a.posdep, nparams=int(A.numel() + off.numel()), test=te,
           coverage_topK=cov, err_set=float((pk[err] == mte[err]).mean()), c1_test=float((c1(Xte) == mte).mean()))
print(json.dumps(res))
with open("residual_ceiling.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
