"""Hugo: check post numbers in _notes/max-pool-survives.md that my rebuild did not already cover.
Own max14 / last14 fits (C=0.01, my CV pick). Lines refer to the draft at blog 'max-pool-survives.md'."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
P = "/Users/esben/work/theory-1019-max-pool-survives/kran-research/projects/max-pool-survives/data"
split = json.load(open(f"{P}/split.json")); N = len(split)
isF = np.array([1 - r["true"] for r in split]); sp = np.array([r["split"] for r in split])
ct = np.load(f"{P}/clean_tokens.npz"); off = ct["tok_off"]; tst = ct["tok_stmt"]; tid = ct["tok_id"]
X = {"clean": ct["tok_L14"]}
for s in range(3):
    X[s] = np.load(f"{P}/runs/lam0__seed{s}__tokens.npz")["tok_L14"]
pr = np.load(f"{P}/probe.npz"); wf, bf = pr["w"], float(pr["b"])

def fit(mask):
    idx = np.where(mask)[0]; mu = X["clean"][idx].mean(0); sd = X["clean"][idx].std(0) + 1e-6
    m = LogisticRegression(C=0.01, max_iter=5000).fit((X["clean"][idx] - mu) / sd, isF[tst][idx])
    return m.coef_[0] / sd, m.intercept_[0] - (mu / sd) @ m.coef_[0]
probe_tok = sp[tst] == "probe"
last = np.zeros(len(tst), bool); last[off[1:] - 1] = True
w14, b14 = fit(probe_tok); wl, bl = fit(probe_tok & last)
rel = np.full(len(tst), -999); twin = np.full(N, -1)
by = {}
for i, r in enumerate(split): by.setdefault((r["city"], r["src"]), []).append(i)
for pair in by.values():
    if len(pair) != 2: continue
    a, b = pair; twin[a], twin[b] = b, a
    ta, tb = tid[off[a]:off[a + 1]], tid[off[b]:off[b + 1]]; k = 0
    while k < min(len(ta), len(tb)) and ta[k] == tb[k]: k += 1
    for i in pair: rel[off[i]:off[i + 1]] = np.arange(off[i + 1] - off[i]) - k
ans = rel >= 0; ev = sp[tst] == "eval"; y = isF[tst]
refT = np.where((sp == "refit") & (isF == 0))[0]; evF = np.where((sp == "eval") & (isF == 1))[0]
evT = np.where((sp == "eval") & (isF == 0))[0]

# line 72: mean drop of max14 token score, answer tokens false / true, shared tokens
print("L72 drop of token score (clean - attacked), eval split:")
for s in range(3):
    d = (X["clean"] - X[s]) @ w14
    print(f"  seed{s}: answer-false {d[ev & ans & (y == 1)].mean():.2f}  answer-true {d[ev & ans & (y == 0)].mean():.2f}"
          f"  shared {d[ev & ~ans].mean():.2f}")
# line 107: paired comparison, max14
mx = lambda st: np.maximum.reduceat(X[st] @ w14 + b14, off[:-1])
for st in ["clean", 0, 1, 2]:
    m = mx(st); f = evF; t = twin[f]
    print(f"L107 {st}: false>twin {np.mean(m[f] > m[t]):.3f}  tie {np.mean(m[f] == m[t]):.3f}  false<twin {np.mean(m[f] < m[t]):.3f}")
# line 169: first answer token only
for st in ["clean", 0, 1, 2]:
    ts = X[st] @ w14 + b14; msk = rel == 0
    sc = np.full(N, -np.inf); sc[tst[msk]] = ts[msk]
    tt = np.sort(sc[refT])[-2]
    print(f"L169 first-answer-token-only {st}: catch {(sc[evF] > tt).mean():.4f}")
# line 76: correlation of last14 / max14 statement scores with the attacked (frozen) probe score
fz = np.add.reduceat(X["clean"] @ wf, off[:-1]) / np.diff(off) + bf
for name, (w, b, pool) in {"last14": (wl, bl, "last"), "max14": (w14, b14, "max")}.items():
    ts = X["clean"] @ w + b
    sc = ts[off[1:] - 1] if pool == "last" else np.maximum.reduceat(ts, off[:-1])
    ev_i = np.r_[evT, evF]
    within = np.mean([np.corrcoef(sc[ix], fz[ix])[0, 1] for ix in (evT, evF)])
    tokcorr = np.corrcoef(ts[ev], (X["clean"] @ wf)[ev])[0, 1]
    print(f"L76 {name}: statement-score corr with frozen (eval, all) {np.corrcoef(sc[ev_i], fz[ev_i])[0,1]:.3f}  "
          f"within-class mean {within:.3f}  token-level corr {tokcorr:.3f}")
# line 156: bit-for-bit vs 1018 saved scores
for s in range(3):
    a = np.load(f"{P}/runs/lam0__seed{s}.npz")["scores"]; b = np.load(f"{P}/ref1018/lam0__seed{s}.npz")["scores"]
    print(f"L156 seed{s}: array_equal(1019 scores, 1018 scores) = {np.array_equal(a, b)}  maxabs {np.abs(a-b).max():.2e}")
# line 166: held-out half thresholds over many random city halves
cities = np.unique(np.array([r["city"] for r in split])[refT]); cityarr = np.array([r["city"] for r in split])
rng = np.random.default_rng(1); vals = []
for _ in range(200):
    h = rng.permutation(cities)[:50]; rT = refT[np.isin(cityarr[refT], h)]
    vals.append(np.mean([ (mx(s)[evF] > np.sort(mx(s)[rT])[-2]).mean() for s in range(3)]) if False else None)
# cheaper: precompute
M = {s: mx(s) for s in range(3)}
vals = []
for _ in range(200):
    h = rng.permutation(cities)[:50]; rT = refT[np.isin(cityarr[refT], h)]
    vals.append([(M[s][evF] > np.sort(M[s][rT])[-2]).mean() for s in range(3)])
vals = np.array(vals)
print(f"L166 half-line over 200 random halves: share of (half,seed) with catch 0: {np.mean(vals == 0):.3f}; "
      f"max {vals.max():.4f}; mean {vals.mean():.4f}")
