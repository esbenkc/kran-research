"""Vera: pre-data power of the frozen TOST verdict (goal: prompt-on arm, trip|says vs trip|not-says within 10 pts).

Design from the goal: 8 agents x 25 runs in the prompt-on arm. Agent-level heterogeneity in says-rate and trip-rate.
Estimators:
  pooled  = run-level difference of proportions, Wald 90% CI (ignores agents)
  mh      = Mantel-Haenszel risk difference over agents (only agents with both cells), agent-cluster bootstrap 90% CI
TOST at +-10 pts == 90% CI inside (-0.10, +0.10).
Reports P(CI inside band) when the true within-agent gap is 0 (power) and when it is -0.30 (false 'equivalent').
"""
import numpy as np

rng = np.random.default_rng(0)
K, N, BAND = 8, 25, 0.10


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def mh_rd(t1, n1, t0, n0):
    ok = (n1 > 0) & (n0 > 0)
    if ok.sum() == 0:
        return np.nan
    w = n1[ok] * n0[ok] / (n1[ok] + n0[ok])
    return np.sum(w * (t1[ok] / n1[ok] - t0[ok] / n0[ok])) / w.sum()


def sim(say_mean, trip_mean, gap, nsim=400, B=400, sd=1.0):
    res = {"pooled": 0, "mh": 0, "mh_nan": 0, "small_cell": 0}
    for _ in range(nsim):
        ps = expit(logit(say_mean) + sd * rng.standard_normal(K))
        pt = expit(logit(trip_mean) + sd * rng.standard_normal(K))
        say = rng.random((K, N)) < ps[:, None]
        p_trip = np.clip(pt[:, None] + gap * (say - say.mean(1, keepdims=True)), 0, 1)
        trip = rng.random((K, N)) < p_trip
        n1 = say.sum(1); n0 = N - n1
        t1 = (trip & say).sum(1); t0 = (trip & ~say).sum(1)
        if n0.sum() < 10 or n1.sum() < 10:
            res["small_cell"] += 1
        # pooled Wald
        N1, N0 = n1.sum(), n0.sum()
        if N1 > 0 and N0 > 0:
            a, b = t1.sum() / N1, t0.sum() / N0
            se = np.sqrt(a * (1 - a) / N1 + b * (1 - b) / N0) + 1e-12
            d = a - b
            if d - 1.645 * se > -BAND and d + 1.645 * se < BAND:
                res["pooled"] += 1
        # MH + cluster bootstrap
        est = mh_rd(t1, n1, t0, n0)
        if np.isnan(est):
            res["mh_nan"] += 1
            continue
        idx = rng.integers(0, K, (B, K))
        bs = np.array([mh_rd(t1[i], n1[i], t0[i], n0[i]) for i in idx])
        bs = bs[~np.isnan(bs)]
        lo, hi = np.quantile(bs, [0.05, 0.95])
        if lo > -BAND and hi < BAND:
            res["mh"] += 1
    return {k: v / nsim for k, v in res.items()}


if __name__ == "__main__":
    print(f"K={K} agents, N={N} runs each, band +-{BAND}")
    print("say  trip  gap   P(pooled TOST pass)  P(MH-cluster TOST pass)  P(MH undefined)  P(a cell <10 runs)")
    for say in (0.3, 0.5, 0.8, 0.95):
        for trip in (0.2, 0.5, 0.8):
            for gap in (0.0, -0.3):
                r = sim(say, trip, gap)
                print(f"{say:.2f} {trip:.2f} {gap:+.1f}   {r['pooled']:.3f}               {r['mh']:.3f}                    {r['mh_nan']:.3f}            {r['small_cell']:.3f}")
