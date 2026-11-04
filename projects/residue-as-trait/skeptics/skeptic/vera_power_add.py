"""Variant: additive carry-over statistic D_add (agent offset on top of the held-out task level).
offset_a = logit(rate of a on other scenarios) - logit(rate of the other 4 agents on other scenarios)
p_add(a,t) = sigmoid(logit(task rate of t among other 4 agents) + offset_a)
D_add = mean over runs of (ll_task - ll_add). Same null (agent labels within scenario x arm).
Rule tested: CONF iff min(D_add, D_base?) ... reported both: rule ADD (D_add alone) and rule MIN3 = min(D_base, D_task, ...) is v2.1."""
import numpy as np, json
from vera_power_v21 import world, A, lg
def dadd(y):
    A_, T = y.shape[:2]; c = y.sum(axis=(2, 3)); n = y.shape[2]*y.shape[3]
    ta, tt, tot = c.sum(1), c.sum(0), c.sum()
    ka = ta[:, None] - c; na = n*(T-1)                       # agent a, other scenarios
    ko = (tot - tt)[None, :] - ka; no = n*(T-1)*(A_-1)        # other agents, other scenarios
    off = lg((ka+1)/(na+2)) - lg((ko+1)/(no+2))
    pt = (tt[None, :] - c + 1)/(n*(A_-1)+2)
    pad = 1/(1+np.exp(-(lg(pt)+off)))
    L = lambda p: -(c*np.log(p) + (n-c)*np.log(1-p))
    return (L(pt) - L(pad)).sum()/y.size
def null(y, rng, nperm):
    A_, T, S, R_ = y.shape; z = y.transpose(1, 3, 0, 2).reshape(T, R_, A_*S); out = np.empty(nperm)
    for i in range(nperm):
        p = np.argsort(rng.random(z.shape), -1)
        out[i] = dadd(np.take_along_axis(z, p, -1).reshape(T, R_, A_, S).transpose(2, 0, 3, 1))
    return out
rng = np.random.default_rng(5); res = {}
for T, S in ((8, 5), (4, 10)):
  for tau in (0.7, 1.5):
    for name, kw in (("agent sd0", dict(s_int=0)), ("sd1.0", dict(s_int=1.0)), ("sd1.14", dict(s_int=1.14)), ("sd1.5", dict(s_int=1.5)),
                     ("sd2.5", dict(s_int=2.5)), ("null sd0", dict(s_int=0, trait=False)), ("null sd1.5", dict(s_int=1.5, trait=False))):
        lab = []; ds = []
        for _ in range(300):
            y = world(rng, T, S, tau=tau, **kw); d = dadd(y); q = np.quantile(null(y, rng, 300), .95)
            lab.append("CONF" if (d >= 0.020 and d > q) else "REJ" if d <= 0.005 else "UNC"); ds.append(d)
        lab = np.array(lab); k = f"{T}x{S} tau{tau} {name}"
        res[k] = dict(P_conf=float((lab=="CONF").mean()), P_rej=float((lab=="REJ").mean()), D_med=float(np.median(ds)))
        print(f"{k:26s} YES {res[k]['P_conf']:.2f} NO {res[k]['P_rej']:.2f} UNC {1-res[k]['P_conf']-res[k]['P_rej']:.2f}  D_add med {res[k]['D_med']:+.3f}", flush=True)
json.dump(res, open("vera_power_add.json", "w"), indent=1)
