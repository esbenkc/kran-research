"""Hugo (skeptic2) independent checker. Own float64 numpy forward from m16_10.pth.
Dev seeds 7000-7099 only; holdout 432432 is never used.
Run: cd swarm-algzoo-432 && .venv/bin/python skeptic2/hugo_check.py
"""
import json, math, sys
import numpy as np
import torch

sd = torch.load("m16_10.pth", map_location="cpu")
WI = sd["rnn.weight_ih_l0"].double().numpy()[:, 0]
WH = sd["rnn.weight_hh_l0"].double().numpy()
WO = sd["linear.weight"].double().numpy()
assert WI.size + WH.size + WO.size == 432


def logits(wi, wh, wo, X):
    h = np.zeros((X.shape[0], 16))
    for t in range(X.shape[1]):
        h = np.maximum(0.0, X[:, t:t + 1] * wi[None, :] + h @ wh.T)
    return h @ wo.T


def pred(w, X):
    return logits(*w, X).argmax(1)


def sample(seed, n):
    return np.random.default_rng(seed).standard_normal((n, 10))


out = {}
# 0. forward check vs torch
X = sample(7000, 20000)
rnn = torch.nn.RNN(1, 16, bias=False, batch_first=True, nonlinearity="relu")
rnn.weight_ih_l0.data = sd["rnn.weight_ih_l0"].double(); rnn.weight_hh_l0.data = sd["rnn.weight_hh_l0"].double()
rnn = rnn.double()
with torch.no_grad():
    hT = rnn(torch.tensor(X)[:, :, None])[0][:, -1]
    lt = (hT @ sd["linear.weight"].double().T).numpy()
out["torch_max_abs_diff"] = float(np.abs(lt - logits(WI, WH, WO, X)).max())

# 1. base facts on dev seed 7001, 1e6
X = sample(7001, 1_000_000)
P0 = pred((WI, WH, WO), X)
order = np.argsort(-X, 1)
out["acc_vs_true_2nd"] = float((P0 == order[:, 1]).mean())


# 2. noise floor: multiplicative N(0, s) noise on all 432 raw weights
def floor(sigma, ndraw, X, P0, seed):
    r = np.random.default_rng(seed)
    a = []
    for _ in range(ndraw):
        w = [W * (1 + sigma * r.standard_normal(W.shape)) for W in (WI, WH, WO)]
        a.append((pred(w, X) == P0).mean())
    return np.array(a)


Xf = X[:200_000]; Pf = P0[:200_000]
f01 = floor(1e-3, 200, Xf, Pf, 7010)
out["floor_0.1pct_200draws_2e5"] = dict(mean=f01.mean(), median=float(np.median(f01)), sd=f01.std(ddof=1),
                                         se_mean=f01.std(ddof=1) / math.sqrt(len(f01)), min=f01.min(), max=f01.max(),
                                         q10=float(np.quantile(f01, .1)), q90=float(np.quantile(f01, .9)),
                                         frac_draws_below_0p98671=float((f01 < 0.98671).mean()))
# replicate the prereg procedure (20 draws x 1e6) on two dev noise seeds
out["floor_0.1pct_20draws_1e6"] = [float(floor(1e-3, 20, X, P0, s).mean()) for s in (7020, 7021)]
for s_ in (3e-4, 3e-3):
    out[f"floor_{s_*100:g}pct_40draws_2e5"] = float(floor(s_, 40, Xf, Pf, 7030).mean())


# 3. gauge: W_hi -> sign, W_oh row 0 -> 0 (row shift). Then rounding to d significant digits.
def gauge(wi, wh, wo, ref=0):
    s = 1.0 / np.abs(wi)                      # h_i' = s_i h_i
    wi2 = np.sign(wi)
    wh2 = wh * s[:, None] / s[None, :]
    wo2 = wo / s[None, :]
    wo2 = wo2 - wo2[ref][None, :]
    return wi2, wh2, wo2


def rsig(a, d):
    a = np.asarray(a, float); o = np.zeros_like(a); nz = a != 0
    e = np.floor(np.log10(np.abs(a[nz])))
    o[nz] = np.round(a[nz] / 10 ** (e - d + 1)) * 10 ** (e - d + 1)
    return o


G = gauge(WI, WH, WO)
assert (pred(G, Xf) == Pf).all()
Xd = sample(7002, 1_000_000)
Pd = pred((WI, WH, WO), Xd)
rows = {}
for ref in range(10):
    g = gauge(WI, WH, WO, ref)
    rows[ref] = {d: float((pred((g[0], rsig(g[1], d), rsig(g[2], d)), Xd) == Pd).mean()) for d in (2, 3, 4)}
out["round_all400_by_ref_row"] = rows

# rounding error size vs the 0.1% noise
g = G
rel = np.concatenate([(rsig(g[1], 3) - g[1])[g[1] != 0] / g[1][g[1] != 0],
                      (rsig(g[2], 3) - g[2])[g[2] != 0] / g[2][g[2] != 0]])
out["round3_rms_rel_err"] = float(np.sqrt((rel ** 2).mean()))
# Gaussian mult noise at that rms, on the 400 gauge numbers only
r = np.random.default_rng(7040); a = []
sig = out["round3_rms_rel_err"]
for _ in range(40):
    a.append((pred((g[0], g[1] * (1 + sig * r.standard_normal(g[1].shape)), g[2] * (1 + sig * r.standard_normal(g[2].shape))), Xf) == Pf).mean())
out["gauss_noise_at_round3_rms_40draws"] = float(np.mean(a))
# and 0.1% noise applied in the gauge (gauge numbers only, W_hi exact) for comparison
r = np.random.default_rng(7041); a = []
for _ in range(40):
    a.append((pred((g[0], g[1] * (1 + 1e-3 * r.standard_normal(g[1].shape)), g[2] * (1 + 1e-3 * r.standard_normal(g[2].shape))), Xf) == Pf).mean())
out["floor_0.1pct_gauge400_40draws"] = float(np.mean(a))


# 4. saved candidate nets: literal count, digits, agreement on dev 7002 (1e6), bits
def load(f):
    d = np.load(f); ks = list(d)
    return [d[k].astype(float) for k in ks]


def ndig(a):
    a = a[a != 0]
    return np.array([next(d for d in range(1, 17) if abs(rsig(v, d) - v) <= 1e-12 * abs(v)) for v in a.ravel()])


def bits(wh, wo, digits):
    """sign + mantissa log2(9*10^(d-1)) + empirical exponent entropy + per-entry digit-level entropy
    (only if mixed) + log2 C(400, k) mask. W_hi = +-1 or 0: 16*log2(3)."""
    v = np.concatenate([wh[wh != 0], wo[wo != 0]])
    k = v.size
    e = np.floor(np.log10(np.abs(v))).astype(int)
    _, c = np.unique(e, return_counts=True); p = c / k
    H_e = float(-(p * np.log2(p)).sum())
    dd = np.asarray(digits)
    mant = float(np.sum(np.log2(9 * 10.0 ** (dd - 1))))
    _, cd = np.unique(dd, return_counts=True); pd = cd / k
    H_d = float(-(pd * np.log2(pd)).sum()) if len(cd) > 1 else 0.0
    mask = (math.lgamma(401) - math.lgamma(k + 1) - math.lgamma(400 - k + 1)) / math.log(2)
    return dict(k=k, sign=k, mant=mant, exp=k * H_e, digit_level=k * H_d, mask=mask, wi=16 * math.log2(3),
                total=k + mant + k * H_e + k * H_d + mask + 16 * math.log2(3))


cands = {"ladder0.99@4d": ("ladder_pruned_0.99.npz", 4), "ladder0.995@3d": ("ladder_pruned_0.995.npz", 3),
         "mixprec_0.989": ("mixprec_0.989.npz", None), "fitted364@3d": ("prune_ft_long_220_144_q3f.npz", None),
         "fitted340@3d": ("prune_ft_long_200_140_q3f.npz", None)}
res = {}
for name, (f, d) in cands.items():
    wi, wh, wo = load(f)
    if d:
        wh, wo = rsig(wh, d), rsig(wo, d)
    dig = np.concatenate([ndig(wh), ndig(wo)])
    agr = float((pred((wi, wh, wo), Xd) == Pd).mean())
    res[name] = dict(literals=int((wh != 0).sum() + (wo != 0).sum()), wi_nonzero=int((wi != 0).sum()),
                     wi_vals=sorted(set(np.round(wi, 9).tolist())), digit_hist=np.bincount(dig).tolist(),
                     agree_dev7002=agr, bits=bits(wh, wo, dig))
g = G
res["all400@3d"] = dict(literals=400, agree_dev7002=rows[0][3], bits=bits(rsig(g[1], 3), rsig(g[2], 3), [3] * 400 if True else None))
out["cands"] = res


def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


json.dump(conv(out), open("skeptic2/hugo_check.json", "w"), indent=1)
print(json.dumps(conv(out), indent=1))
