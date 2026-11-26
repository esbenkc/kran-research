# Power of the v2 status rules, before the stamp (Tove #31383 asks: true R 0.3/0.5/0.75/1.0).
# Model: ln T[arm,seed] = ln(296*mult[arm]) + v[seed] (cue-vector effect, shared by all arms of a seed) + sd*eps.
# Parent k4 noise: T_dense 296/280/304 (sd_log 0.042, one vector); 3 fresh vectors 296/296/304.
# Rules:  I = Iris #31384 FINAL (CONF: every ref in {FRESH,SHUF,COIN} has upper <= ln .5; REJ: SHUF or FRESH lower > ln .5)
#         F = Felix #31386 (ctrl* = control with smaller geomean T; paired over MAIN seeds; CONF upper<ln.5, REJ lower>ln.5)
# Ratios: paired by seed where the ref arm has that seed, else divided by the ref arm's geomean (as frozen text says).
import numpy as np, sys
from scipy import stats
rng=np.random.default_rng(2); N=100000; L5=np.log(.5)
def sim(R, sd, sdv, nF, nC, coinmult=1.0, freshmult=1.0):
    S=5; v=sdv*rng.standard_normal((N,S))
    e=lambda n: sd*rng.standard_normal((N,n))
    main=np.log(296*R)+v+e(S); shuf=np.log(296)+v+e(S)
    coin=np.log(296*coinmult)+v[:,:nC]+e(nC); fresh=np.log(296*freshmult)+v[:,:nF]+e(nF)
    t=stats.t.ppf(.975,S-1)
    def iv(ref,n):
        if n==S: d=main-ref
        else:
            gm=ref.mean(1,keepdims=True); d=np.concatenate([main[:,:n]-ref,main[:,n:]-gm],1)
        m=d.mean(1); h=t*d.std(1,ddof=1)/np.sqrt(S); return m-h,m+h
    lo_s,hi_s=iv(shuf,S); lo_c,hi_c=iv(coin,nC); lo_f,hi_f=iv(fresh,nF)
    # port gate on FRESH median (296*freshmult*exp(noise)) in [266,326]
    gate=(np.exp(np.median(fresh,1))>=266)&(np.exp(np.median(fresh,1))<=326)
    I_C=(hi_s<=L5)&(hi_c<=L5)&(hi_f<=L5)&gate; I_R=((lo_s>L5)|(lo_f>L5))&~I_C
    pick_coin=coin.mean(1)<shuf.mean(1)
    lo=np.where(pick_coin,lo_c,lo_s); hi=np.where(pick_coin,hi_c,hi_s)
    F_C=(hi<L5)&gate; F_R=(lo>L5)&~F_C
    return I_C.mean(),I_R.mean(),F_C.mean(),F_R.mean(),gate.mean()
print("seeds MAIN/SHUF 5, COIN nC, FRESH nF; P(CONFIRMED), P(REJECTED) for rule I (Iris final) and rule F (Felix ctrl*)")
for (nF,nC) in ((3,5),(3,3),(5,5)):
  for sd,sdv in ((0.042,0.0),(0.08,0.10),(0.15,0.10)):
    for R in (0.3,0.4,0.45,0.5,0.515,0.6,0.75,1.0):
      a=sim(R,sd,sdv,nF,nC)
      print(f"nF {nF} nC {nC} sd {sd:.3f} sdv {sdv:.2f} R {R:.3f} | I: C {a[0]:.3f} R {a[1]:.3f} | F: C {a[2]:.3f} R {a[3]:.3f} | gate {a[4]:.3f}")
print("stress: COIN 1.3x slower than SHUF, FRESH 1.1x slower (sd .08 sdv .10, nF 3 nC 5)")
for R in (0.3,0.45,0.5,0.6,1.0):
    a=sim(R,0.08,0.10,3,5,1.3,1.1); print(f"R {R} | I: C {a[0]:.3f} R {a[1]:.3f} | F: C {a[2]:.3f} R {a[3]:.3f} | gate {a[4]:.3f}")
