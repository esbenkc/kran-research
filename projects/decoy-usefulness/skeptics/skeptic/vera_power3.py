"""Full frozen title rule (Tove #10143/#10146): YES = run-level 95% low > 0 AND >=4/5 models d>0;
NO = 95% high < 0 OR whole CI inside [-0.15, +0.15]; else NS. Precedence variants: (A) YES checked first, (B) NO-equal first.
Same 1028-diagonal worlds as vera_power2.py. Also reports mean CI width."""
import numpy as np
rng = np.random.default_rng(2)
UR = np.array([.800, .933, 1.0, .900, .800]); LN = np.array([.000, .000, .233, .333, .033])
N, B, W = 20, 2000, 500
def eff(c):
    U = (c[...,0]+c[...,1]-c[...,2]-c[...,3])/2; R = (c[...,0]+c[...,2]-c[...,1]-c[...,3])/2; return U, R
for lf, lr in [(.8,.3),(.7,.4),(.65,.45),(.6,.5),(.55,.55),(.5,.6),(.4,.7)]:
    cnt = {"A":np.zeros(4), "B":np.zeros(4)}; widths = []
    for w in range(W):
        p = np.stack([UR, lf*UR+(1-lf)*LN, lr*UR+(1-lr)*LN, LN], 1)
        y = rng.binomial(1, np.repeat(p[...,None], N, -1))
        U, R = eff(y.mean(-1)); d = U - R
        idx = rng.integers(0, N, (B,)+y.shape)
        rb = np.take_along_axis(np.broadcast_to(y, (B,)+y.shape), idx, -1).mean(-1)
        Ub, Rb = eff(rb); lo, hi = np.percentile((Ub-Rb).mean(-1), [2.5, 97.5]); widths.append(hi-lo)
        yes = lo > 0 and (d > 0).sum() >= 4; eq = lo > -.15 and hi < .15; no = hi < 0 or eq
        a = "YES" if yes else ("NO" if no else "NS")
        b = "NO" if no else ("YES" if yes else "NS")
        for k, v in (("A", a), ("B", b)):
            cnt[k][["YES","NO","NS"].index(v)] += 1
        cnt["A"][3] += yes and eq
    true = ((lf-lr)*(UR-LN)).mean()
    print(f"true diff {true:+.3f} meanCIwidth {np.mean(widths):.3f}: YES-first YES {cnt['A'][0]/W:.3f} NO {cnt['A'][1]/W:.3f} NS {cnt['A'][2]/W:.3f} "
          f"| NO-first YES {cnt['B'][0]/W:.3f} NO {cnt['B'][1]/W:.3f} | YES&equal overlap {cnt['A'][3]/W:.3f}")
