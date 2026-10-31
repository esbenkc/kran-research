"""Vera: alternative verdict estimators vs Felix's rule, K=5 models, pre-data.
A  = Felix: unweighted mean of per-model gaps, two-level bootstrap (models, then runs).
B  = MH risk difference (weights n1*n0/(n1+n0)), stratified run bootstrap within model (claim about THESE models).
C  = MH risk difference, model-cluster bootstrap.
Yes = 90% CI inside (-0.105, 0.105); No = CI fully outside; else Not settled. Qualifying model: >=5 runs per say-cell; <3 -> NS.
"""
import numpy as np

rng = np.random.default_rng(2)
BAND = 0.105


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def mh(say, trip, weighted=True):
    num = den = 0.0
    q = 0
    gaps = []
    for s, t in zip(say, trip):
        n1, n0 = s.sum(), (~s).sum()
        if n1 >= 5 and n0 >= 5:
            g = t[s].mean() - t[~s].mean()
            w = n1 * n0 / (n1 + n0) if weighted else 1.0
            num += w * g
            den += w
            q += 1
    return num / den if q >= 3 else np.nan


def verdict(lo, hi):
    if lo > -BAND and hi < BAND:
        return "YES"
    if hi < -BAND or lo > BAND:
        return "NO"
    return "NS"


def one(K, N, say_mean, trip_mean, gap, B=300, sd=1.0):
    ps = expit(logit(say_mean) + sd * rng.standard_normal(K))
    pt = expit(logit(trip_mean) + sd * rng.standard_normal(K))
    say = rng.random((K, N)) < ps[:, None]
    p = np.clip(pt[:, None] + gap * (say - say.mean(1, keepdims=True)), 0, 1)
    trip = rng.random((K, N)) < p
    if np.isnan(mh(say, trip)):
        return ("deg",) * 3
    out = {}
    for name, w, mod, runs in (("A", False, True, True), ("B", True, False, True), ("C", True, True, False)):
        bs = []
        for _ in range(B):
            mi = rng.integers(0, K, K) if mod else np.arange(K)
            s, t = say[mi], trip[mi]
            if runs:
                ri = rng.integers(0, N, (K, N))
                s, t = np.take_along_axis(s, ri, 1), np.take_along_axis(t, ri, 1)
            v = mh(s, t, w)
            if not np.isnan(v):
                bs.append(v)
        if len(bs) < B / 2:
            out[name] = "deg"
            continue
        lo, hi = np.quantile(bs, [0.05, 0.95])
        out[name] = verdict(lo, hi)
    return out["A"], out["B"], out["C"]


if __name__ == "__main__":
    nsim = 250
    K = 5
    for N in (100, 150):
        for say in (0.5, 0.8):
            for gap in (0.0, -0.105, -0.30):
                res = [one(K, N, say, 0.5, gap) for _ in range(nsim)]
                line = f"K={K} N={N} say {say:.2f} gap {gap:+.3f}:"
                for j, nm in enumerate("ABC"):
                    col = [r[j] for r in res]
                    line += f"  {nm}: YES {col.count('YES')/nsim:.2f} NO {col.count('NO')/nsim:.2f}"
                print(line, flush=True)
