# Vera pre-data power sim: 5 agents, base rates from decoy-usefulness useless_real_explicit (n=20 each),
# common log-odds shift d in the test arm; stratified (agent-weighted equal) drop with 95% normal CI.
import numpy as np, itertools
rng=np.random.default_rng(1105)
base=np.array([0.05,0.30,0.95,0.00,0.55])  # haiku, g3.1, g3.5, mini, nano
lg=lambda p: np.log(p/(1-p)); ex=lambda x:1/(1+np.exp(-x))
def sim(d, n=40, reps=4000, drift_sd=0.5):
    out=[]
    for _ in range(reps):
        b=ex(lg(np.clip(base,0.02,0.98))+rng.normal(0,drift_sd,5))
        t=ex(lg(b)-d)
        xp=rng.binomial(n,b); xt=rng.binomial(n,t)
        pp=xp/n; pt=xt/n
        drop=(pp-pt).mean()
        var=((pp*(1-pp)+pt*(1-pt))/n).sum()/25
        se=np.sqrt(var+1e-12)
        out.append((drop, drop-1.96*se, drop+1.96*se, (b-t).mean(), ((pp-pt)>=0.15).sum()))
    return np.array(out)
print("d  truedrop  mean_hw  P(YES lo>0)  P(YES lo>5)  P(NO hi<15)  P(NO hi<10)  P(NS|lo>0,hi<15)  P(>=3 agents drop15)")
for d in [0,0.25,0.5,0.75,1.0,1.33,2.0]:
    r=sim(d)
    yes0=(r[:,1]>0); yes5=(r[:,1]>0.05); no15=(r[:,2]<0.15); no10=(r[:,2]<0.10)
    print(f"{d:4.2f} {r[:,3].mean():7.3f} {((r[:,2]-r[:,1])/2).mean():7.3f} {yes0.mean():10.2f} {yes5.mean():10.2f} {no15.mean():10.2f} {no10.mean():10.2f} {(~yes0&~no15).mean():10.2f} {(r[:,4]>=3).mean():8.2f}")
