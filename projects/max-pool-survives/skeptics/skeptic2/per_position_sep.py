"""Hugo: check Bruno's per-position separation (first answer token 4.56 -> ~0.92, next 5.13 -> ~0.42) with
my own max14 fit (C=0.01, as picked by my CV in rebuild.py). Separation at relative position r (r=0 = first
token that differs from the twin statement) = (mean false - mean true of the token score) / pooled clean
within-class SD at that same position. Eval split."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
P = "/Users/esben/work/theory-1019-max-pool-survives/kran-research/projects/max-pool-survives/data"
split = json.load(open(f"{P}/split.json")); N = len(split)
isF = np.array([1 - r["true"] for r in split]); sp = np.array([r["split"] for r in split])
ct = np.load(f"{P}/clean_tokens.npz"); off = ct["tok_off"]; tst = ct["tok_stmt"]; tid = ct["tok_id"]
X = {"clean": ct["tok_L14"]}
for s in range(3):
    X[s] = np.load(f"{P}/runs/lam0__seed{s}__tokens.npz")["tok_L14"]
idx = np.where(sp[tst] == "probe")[0]
mu = X["clean"][idx].mean(0); sd = X["clean"][idx].std(0) + 1e-6
m = LogisticRegression(C=0.01, max_iter=5000).fit((X["clean"][idx] - mu) / sd, isF[tst][idx])
w = m.coef_[0] / sd
# relative position to the first differing token, per statement
rel = np.full(len(tst), -999)
by = {}
for i, r in enumerate(split):
    by.setdefault((r["city"], r["src"]), []).append(i)
for pair in by.values():
    if len(pair) != 2: continue
    a, b = pair; ta, tb = tid[off[a]:off[a + 1]], tid[off[b]:off[b + 1]]
    k = 0
    while k < min(len(ta), len(tb)) and ta[k] == tb[k]: k += 1
    for i in pair:
        rel[off[i]:off[i + 1]] = np.arange(off[i + 1] - off[i]) - k
ev = sp[tst] == "eval"; y = isF[tst]
out = {}
for r_ in (-2, -1, 0, 1):
    msk = ev & (rel == r_)
    c = X["clean"][msk] @ w; yy = y[msk]
    sdw = np.sqrt(((c[yy == 1] - c[yy == 1].mean()) ** 2).sum() + ((c[yy == 0] - c[yy == 0].mean()) ** 2).sum()) / np.sqrt(len(c) - 2)
    row = []
    for st in ["clean", 0, 1, 2]:
        v = X[st][msk] @ w
        row.append(round(float((v[yy == 1].mean() - v[yy == 0].mean()) / sdw), 3))
    out[r_] = row
    print(f"rel pos {r_:+d} (n={msk.sum()}): clean, seed0, seed1, seed2 = {row}")
json.dump({str(k): v for k, v in out.items()}, open(__file__.replace(".py", ".json"), "w"))
