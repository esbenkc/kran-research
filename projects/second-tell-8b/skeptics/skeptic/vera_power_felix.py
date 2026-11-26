# Felix's proposed frozen rule (#31377), simulated. Per-run log-noise sd (parent T_dense at k4: sd_log 0.042;
# 3 fresh vectors 296/296/304). MAIN 5, SHUF 5 (paired by seed), COIN 3, FRESH 3 (ratio to arm geo-mean).
import numpy as np
from scipy import stats
rng=np.random.default_rng(1); N=40000; L5=np.log(0.5)
def run(R, sd, sdv=0.0):
    out={'C':0,'R':0,'U':0}
    t5=stats.t.ppf(0.975,4)
    for _ in range(N):
        v=sdv*rng.standard_normal(5)                      # cue-vector effect per seed (shared u_B)
        main=np.log(296*R)+v+sd*rng.standard_normal(5)
        shuf=np.log(296)+v+sd*rng.standard_normal(5)
        coin=np.log(296)+v[:3]+sd*rng.standard_normal(3)
        fresh=np.log(296)+v[:3]+sd*rng.standard_normal(3)
        ivs={}
        for k,ref in (('shuf',main-shuf),('coin',main-coin.mean()),('fresh',main-fresh.mean())):
            m=ref.mean(); h=t5*ref.std(ddof=1)/np.sqrt(5); ivs[k]=(m-h,m+h)
        allseed=np.all(main[:,None]<=np.r_[shuf.max(),coin.max(),fresh.max()]+L5)  # loose reading
        allseed=np.all(main<=shuf+L5) and np.all(main<=coin.mean()+L5) and np.all(main<=fresh.mean()+L5)
        if all(ivs[k][1]<=L5 for k in ivs) and allseed: out['C']+=1
        elif ivs['shuf'][0]>L5 or ivs['fresh'][0]>L5: out['R']+=1
        else: out['U']+=1
    return {k:v/N for k,v in out.items()}
for sd,sdv in ((0.042,0.0),(0.08,0.0),(0.08,0.10)):
    for R in (1.0,0.75,0.6,0.515,0.5,0.45,0.4,0.33):
        r=run(R,sd,sdv); print(f"sd {sd} sdv {sdv} true ratio {R:.3f}: CONF {r['C']:.3f} REJ {r['R']:.3f} UNC {r['U']:.3f}")
