# Vera: simulate the WHOLE frozen rule of tell-chain-8b PREREG draft (Iris #32063), before any data.
# Status: both refs (CTRL-D, FRESH-D): CONF if both upper < ln .55; REJ if either lower > ln .55; censor >1 -> ref out;
# PORT GATE: FRESH-D median T in [266, 326] else at most UNC. Prediction clause 1: geo R_D vs CTRL-D <= .55;
# clause 2: R_B (parent, fixed) > R_C > R_D strictly in >= 4 of 5 seeds; held needs both and status != REJ.
import numpy as np
from scipy import stats
rng = np.random.default_rng(112701)
RB = np.array([0.857, 0.714, 0.824, 0.706, 0.750])     # parent per-seed R_B vs COIN (known fact)
CAP, n, NS = 1536, 5, 20000
tc = stats.t.ppf(.975, n-1); bar = np.log(.55)
def grid(t): return np.minimum(CAP, np.maximum(0, 8*np.ceil(t/8)))
def draw(mu, sd): return grid(mu*np.exp(rng.standard_normal((NS, n))*sd))
def ci(a, b):
    d = np.log(np.maximum(a, 8)) - np.log(np.maximum(b, 8)); m = d.mean(1); h = tc*d.std(1, ddof=1)/np.sqrt(n)
    return m, m-h, m+h
def run(RC, RD, drift=1.0, sdm=0.085, sdf=0.07, sdc=0.04, mu=293.):
    fD = draw(mu, sdf); cD = draw(mu*drift, sdc); mD = draw(mu*drift*RD, sdm)
    cC = draw(mu*drift, sdc); mC = draw(mu*drift*RC, sdm)
    port = (np.median(fD, 1) >= 266) & (np.median(fD, 1) <= 326)
    conf = port.copy(); rej = np.zeros(NS, bool)
    for r in (cD, fD):
        ok = ((r >= CAP).sum(1) <= 1) & ((mD >= CAP).sum(1) <= 1)
        m, lo, hi = ci(mD, r); conf &= ok & (hi < bar); rej |= ok & port & (lo > bar)
    rej &= ~conf
    m, _, _ = ci(mD, cD); c1 = m <= bar
    rc = np.maximum(mC, 8)/np.maximum(cC, 8); rd = np.maximum(mD, 8)/np.maximum(cD, 8)
    c2 = ((RB > rc) & (rc > rd)).sum(1) >= 4
    held = c1 & c2 & ~rej
    return dict(CONF=conf.mean(), REJ=rej.mean(), UNC=1-conf.mean()-rej.mean(), port=port.mean(), held=held.mean())
print("model    drift  R_C   R_D   CONF   REJ    UNC   portOK held")
for drift in (1.0, 0.9, 1.1):
    for name, RC, RD in (("mult", .548, .405), ("mult.45", .6, .45), ("mid.50", .62, .50), ("edge.55", .64, .55),
                         ("edge.60", .67, .60), ("flat", .74, .74), ("vanish", .85, .95)):
        o = run(RC, RD, drift)
        print(f"{name:8s} {drift:4.1f} {RC:5.3f} {RD:5.3f}  {o['CONF']:.3f}  {o['REJ']:.3f}  {o['UNC']:.3f}  {o['port']:.3f}  {o['held']:.3f}")
print("-- noisier learners (sdm .15, sdf .10, sdc .08), drift 1.0")
for name, RC, RD in (("mult", .548, .405), ("flat", .74, .74), ("edge.60", .67, .60)):
    o = run(RC, RD, 1.0, .15, .10, .08)
    print(f"{name:8s}  {o['CONF']:.3f}  {o['REJ']:.3f}  {o['UNC']:.3f}  port {o['port']:.3f}  held {o['held']:.3f}")
