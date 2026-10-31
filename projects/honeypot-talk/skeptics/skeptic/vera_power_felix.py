"""Vera: power of Felix's verdict rule (#10610 D), pre-data.
K models x N prompt-on runs. Stat = mean over qualifying models (>=5 runs in each say-cell) of
trip|says - trip|not-says. 90% CI by two-level bootstrap (models, then runs within model).
Yes = CI inside (-0.105, 0.105); No = CI fully outside; else Not settled; <3 qualifying models -> Not settled.
"""
import numpy as np

rng = np.random.default_rng(1)
BAND = 0.105


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def stat(say, trip):
    gaps = []
    for s, t in zip(say, trip):
        if s.sum() >= 5 and (~s).sum() >= 5:
            gaps.append(t[s].mean() - t[~s].mean())
    return np.mean(gaps) if len(gaps) >= 3 else np.nan, len(gaps)


def one(K, N, say_mean, trip_mean, gap, B=300, sd=1.0):
    ps = expit(logit(say_mean) + sd * rng.standard_normal(K))
    pt = expit(logit(trip_mean) + sd * rng.standard_normal(K))
    say = rng.random((K, N)) < ps[:, None]
    p = np.clip(pt[:, None] + gap * (say - say.mean(1, keepdims=True)), 0, 1)
    trip = rng.random((K, N)) < p
    est, q = stat(say, trip)
    if np.isnan(est):
        return "NS-deg"
    bs = []
    for _ in range(B):
        mi = rng.integers(0, K, K)
        ri = rng.integers(0, N, (K, N))
        s = np.take_along_axis(say[mi], ri, 1)
        t = np.take_along_axis(trip[mi], ri, 1)
        v, _ = stat(s, t)
        if not np.isnan(v):
            bs.append(v)
    if len(bs) < B * 0.5:
        return "NS-deg"
    lo, hi = np.quantile(bs, [0.05, 0.95])
    if lo > -BAND and hi < BAND:
        return "YES"
    if hi < -BAND or lo > BAND:
        return "NO"
    return "NS"


if __name__ == "__main__":
    nsim = 300
    for K in (5, 8):
        for N in (50, 100):
            for say in (0.5, 0.8, 0.95):
                for gap in (0.0, -0.10, -0.30):
                    out = [one(K, N, say, 0.5, gap) for _ in range(nsim)]
                    c = {k: out.count(k) / nsim for k in ("YES", "NO", "NS", "NS-deg")}
                    print(f"K={K} N={N:3d} say {say:.2f} gap {gap:+.2f}: YES {c['YES']:.2f} NO {c['NO']:.2f} NS {c['NS']:.2f} degenerate {c['NS-deg']:.2f}", flush=True)
