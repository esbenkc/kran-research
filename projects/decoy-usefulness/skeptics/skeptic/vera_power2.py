"""Power of the frozen TITLE rule on 1028 per-model diagonal rates (strong five).
fit+real = 1028 stateful use, unfit+foreign = 1028 static use (per model). Off-diagonal cells set by scenario
as a mix: fit+foreign = lf*UR + (1-lf)*LN ; unfit+real = lr*UR + (1-lr)*LN.
Usefulness - realism = (lf - lr)*(UR-LN) in expectation. Title rule: YES if 95% low > 0, NO if 95% high < 0.
Estimators: run-level bootstrap stratified by model x cell (primary), model-level t 4 df (sensitivity)."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(1)
UR = np.array([.800, .933, 1.0, .900, .800])   # haiku, g3.1fl, g3.5fl, 5.4-mini, 5.4-nano stateful
LN = np.array([.000, .000, .233, .333, .033])  # static
N, B, W = 20, 2000, 500
def eff(c):
    U = (c[...,0]+c[...,1]-c[...,2]-c[...,3])/2; R = (c[...,0]+c[...,2]-c[...,1]-c[...,3])/2; return U, R
for lf, lr in [(.9,.2),(.8,.3),(.7,.4),(.6,.5),(.55,.55),(.5,.6),(.4,.7),(.2,.9)]:
    res = {"run":[0,0], "t":[0,0]}
    for w in range(W):
        p = np.stack([UR, lf*UR+(1-lf)*LN, lr*UR+(1-lr)*LN, LN], 1)  # cells: UR, UN(fit,foreign), LR(unfit,real), LN
        y = rng.binomial(1, np.repeat(p[...,None], N, -1))
        U, R = eff(y.mean(-1)); d = U - R
        idx = rng.integers(0, N, (B,)+y.shape)
        rb = np.take_along_axis(np.broadcast_to(y, (B,)+y.shape), idx, -1).mean(-1)
        Ub, Rb = eff(rb); db = (Ub-Rb).mean(-1)
        lo, hi = np.percentile(db, [2.5, 97.5])
        h = stats.t.ppf(.975, 4)*d.std(ddof=1)/np.sqrt(5); D = d.mean()
        res["run"][0] += lo > 0; res["run"][1] += hi < 0
        res["t"][0] += D-h > 0; res["t"][1] += D+h < 0
    true = ((lf-lr)*(UR-LN)).mean()
    print(f"fit-share {lf:.2f} real-share {lr:.2f} true diff {true:+.3f}: " +
          " | ".join(f"{k}: YES {v[0]/W:.3f} NO {v[1]/W:.3f} NS {1-(v[0]+v[1])/W:.3f}" for k,v in res.items()))
