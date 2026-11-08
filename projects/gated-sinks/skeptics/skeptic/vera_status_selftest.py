import numpy as np
from scipy import stats
from vera_status import law_status
rng = np.random.default_rng(1)
xs = np.array([.45,.45,.45,.25,.25,.05,.05,.03,.03]); xg = np.array([.05,.05,.05])
agree = 0; N = 2000
for it in range(N):
    slope = rng.choice([0, -0.12, -0.25]); off = rng.choice([0, 0.08])
    ys = 0.7 + slope*xs + rng.normal(0, .02, 9); yg = 0.7 + slope*xg + off + rng.normal(0, .02, 3)
    ms = [dict(run=f"n{i}", arm="x", gated=False, sink=xs[i], recovery=ys[i]) for i in range(9)] + [dict(run=f"g{i}", arm="gated", gated=True, sink=xg[i], recovery=yg[i]) for i in range(3)]
    s = law_status(ms)["status"]
    # vera_sim prereg logic, inline copy for cross-check
    r = stats.linregress(xs, ys); t = stats.t.ppf(.975, 7); lo, hi = r.slope - t*r.stderr, r.slope + t*r.stderr
    st = np.sqrt(((ys-(r.intercept+r.slope*xs))**2).sum()/7); h = t*st*np.sqrt(1/9+(xg-xs.mean())**2/((xs-xs.mean())**2).sum())
    inb = np.abs(yg-(r.intercept+r.slope*xg)) <= np.maximum(max(.03, 2*st), h)
    rej = max(abs(lo),abs(hi))*np.ptp(xs) < .05 or (~inb).sum() >= 2 or lo > 0
    s2 = "REJECTED" if rej else "CONFIRMED" if (hi < 0 and inb.sum() >= 2) else "UNCONFIRMED"
    agree += s == s2
print(f"agree {agree}/{N}")
