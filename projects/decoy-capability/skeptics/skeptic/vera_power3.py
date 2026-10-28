"""Power of rule B (Tove #9002): model-level t-intervals (K-2 df) for I (OLS of d_m on x) and static change
(OLS of u_static on x), times range R; same s6 thresholds. Also the frozen two-level rule for comparison.
Model heterogeneity: per-model offset o_m ~ N(0, sd_o) on both arms, extra gap noise g_m ~ N(0, sd_g) on static."""
import numpy as np
from scipy import stats
from vera_stats import per_model, fitted, summary, boot, status, point

def gen(rng, K, n, drop, sd_o, sd_g, touch=0.6, base=0.6):
    caps = np.linspace(0.1, 0.9, K); eps = []
    for i, cap in enumerate(caps):
        m = f"m{i}"; o = rng.normal(0, sd_o); g = rng.normal(0, sd_g)
        for s in range(20):
            eps.append(dict(model=m, arm="nobait", seed=100+s, touch=0, use=0, score=float(np.clip(cap + rng.normal(0, .15), 0, 1))))
        f = (cap-caps[0])/(caps[-1]-caps[0])
        ps = np.clip(base + o + g - drop*f, 0.02, 0.98); pf = np.clip(base + o, 0.02, 0.98)
        for s in range(n):
            t = int(rng.random() < touch)
            eps.append(dict(model=m, arm="static", seed=s, touch=t, use=int(t and rng.random() < ps), score=0))
            eps.append(dict(model=m, arm="stateful", seed=s, touch=t, use=int(t and rng.random() < pf), score=0))
    return per_model(eps, [f"m{i}" for i in range(K)])

def t_ci(x, y, R):
    r = stats.linregress(x, y); t = stats.t.ppf(0.975, len(x)-2)
    return np.array([r.slope - t*r.stderr, r.slope + t*r.stderr]) * R

def status_B(pm, ms):
    x, us, uf = point(pm, ms); R = x.max()-x.min()
    Ici = t_ci(x, us-uf, R); Sci = t_ci(x, us, R); sc = stats.linregress(x, us).slope*R
    conf = Ici[1] < 0 and Sci[1] < 0 and sc <= -0.10
    rej = Sci[0] > -0.10 or Ici[0] > -0.10
    return "REJECTED" if rej else ("CONFIRMED" if conf else "UNCONFIRMED")

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    reps = 300
    print("K drop sd_o sd_g | rule B (C/R/U) | frozen two-level (C/R/U, 100 reps)")
    for K in (6, 8):
        for drop in (0.0, 0.2, 0.3):
            for sd_o, sd_g in ((0.0, 0.0), (0.1, 0.05)):
                cB = {"CONFIRMED": 0, "REJECTED": 0, "UNCONFIRMED": 0}; cF = dict(cB)
                for r in range(reps):
                    pm = gen(rng, K, 40, drop, sd_o, sd_g); ms = fitted(pm)
                    cB[status_B(pm, ms)] += 1
                    if r < 100:
                        s = summary(pm, ms); cF[status(s, boot(pm, ms, s["R"], B=600, seed=r))] += 1
                fmt = lambda c, n: "/".join(f"{c[k]/n:.2f}" for k in ("CONFIRMED", "REJECTED", "UNCONFIRMED"))
                print(K, drop, sd_o, sd_g, "|", fmt(cB, reps), "|", fmt(cF, 100), flush=True)
