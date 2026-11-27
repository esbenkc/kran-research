# Vera: power of a candidate stage-D rule (written before any data; no data of this run exists).
# Model: ln T = ln(mu_ref) + eps, eps ~ N(0, sd); MAIN T = R * T_ref_true; T observed on 8-update grid
# (first check >= true time). Paired by seed: shared seed effect rho on ln T.
import numpy as np
from scipy import stats
rng = np.random.default_rng(1127)
def grid(t): return np.maximum(8, 8*np.ceil(t/8))
def sim(R_D, sd, n=5, mu=290., L=None, nsim=20000, rho=0.0, Rs=None):
    out = {"CONF":0,"REJ":0,"UNC":0,"mono":0}
    tcrit = stats.t.ppf(.975, n-1); bar = np.log(.55)
    seed = rng.standard_normal((nsim, n))*sd*np.sqrt(rho)
    def T(mult): return grid(mu*mult*np.exp(seed + rng.standard_normal((nsim,n))*sd*np.sqrt(1-rho)))
    main = T(R_D); refs = [T(1.0), T(1.0)]   # coin chain D, fresh D
    conf = np.ones(nsim,bool); rej = np.zeros(nsim,bool)
    for r in refs:
        if L is not None:
            cens = (r > L).sum(1) > 1
        d = np.log(main) - np.log(r); m = d.mean(1); s = d.std(1, ddof=1); hw = tcrit*s/np.sqrt(n)
        conf &= (m+hw < bar); rej |= (m-hw > bar)
    out["CONF"] = conf.mean(); out["REJ"] = (rej & ~conf).mean(); out["UNC"] = 1-out["CONF"]-out["REJ"]
    if Rs is not None:  # monotone B>C>D per seed (ratios vs coin chain), >=4 of 5
        rB = np.log(T(Rs[0]))-np.log(T(1)); rC = np.log(T(Rs[1]))-np.log(T(1)); rDv = np.log(main)-np.log(refs[0])
        mono = ((rB > rC) & (rC > rDv)).sum(1) >= 4; out["mono"] = mono.mean()
    return out
for sd in (0.06, 0.10, 0.15):
    for name, Rs in (("mult", (0.74, 0.548, 0.405)), ("flat", (0.74, 0.74, 0.74)), ("half", (0.74, 0.62, 0.50)),
                     ("edge.55", (0.74, .64, .55)), ("edge.60", (0.74, .67, .60)), ("vanish", (0.74, 0.85, 0.95))):
        o = sim(Rs[2], sd, Rs=Rs)
        print(f"sd {sd:.2f} {name:8s} R_D {Rs[2]:.3f}  CONF {o['CONF']:.3f} REJ {o['REJ']:.3f} UNC {o['UNC']:.3f}  mono>=4/5 {o['mono']:.3f}")
