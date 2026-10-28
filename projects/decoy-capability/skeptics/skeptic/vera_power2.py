"""Compare power: frozen two-level bootstrap vs t-test on d_m slope (K-2 df) vs episode-only bootstrap."""
import numpy as np
from vera_stats import per_model, fitted, summary, d_slope_t, ols_slope

def gen(rng, K, n, drop, touch=0.6, base=0.7):
    caps = np.linspace(0.1, 0.9, K); eps = []
    for i, cap in enumerate(caps):
        m = f"m{i}"
        for s in range(20):
            eps.append(dict(model=m, arm="nobait", seed=100+s, touch=0, use=0, score=float(np.clip(cap + rng.normal(0, .15), 0, 1))))
        f = (cap-caps[0])/(caps[-1]-caps[0])
        for s in range(n):
            t = int(rng.random() < touch)
            eps.append(dict(model=m, arm="static", seed=s, touch=t, use=int(t and rng.random() < base-drop*f), score=0))
            eps.append(dict(model=m, arm="stateful", seed=s, touch=t, use=int(t and rng.random() < base), score=0))
    return per_model(eps, [f"m{i}" for i in range(K)])

def ep_boot(pm, ms, R, B, rng):
    I = []
    x = np.array([pm[m]["nb"].mean() for m in ms])
    for _ in range(B):
        us, uf = [], []
        for m in ms:
            d = pm[m]; j = rng.integers(0, len(d["seeds"]), len(d["seeds"]))
            us.append(d["u_st"][j].sum()/max(d["t_st"][j].sum(),1)); uf.append(d["u_sf"][j].sum()/max(d["t_sf"][j].sum(),1))
        I.append((ols_slope(x, us)-ols_slope(x, uf))*R)
    return np.percentile(I, [2.5, 97.5])

from vera_stats import boot
rng = np.random.default_rng(7)
for K, n, drop in [(6, 40, 0.0), (6, 40, 0.3), (6, 40, 0.5), (8, 40, 0.3)]:
    c = {"two-level excl0": 0, "t d-slope excl0": 0, "episode-only excl0": 0}
    reps = 60
    for r in range(reps):
        pm = gen(rng, K, n, drop); ms = fitted(pm); s = summary(pm, ms)
        b = boot(pm, ms, s["R"], B=800, seed=r)
        c["two-level excl0"] += b["I_ci"][1] < 0
        t = d_slope_t(pm, ms); c["t d-slope excl0"] += t["ci"][1] < 0
        e = ep_boot(pm, ms, s["R"], 800, rng); c["episode-only excl0"] += e[1] < 0
    print(K, n, drop, {k: round(v/reps, 2) for k, v in c.items()}, flush=True)
