# Effect of mini's off-task n (others 60) on the v1.0.2 rule; adds R reachable-zero headline fit combos (pred 0, n=30, law exact = 0 follows).
import numpy as np, vera_sim_rule as v
from vera_sim_off import PAR, VIEW
def run(world,n_mini,nfam=3,nfit_ctx=4,R=1,reps=4000,seed=5,shift=0.2,agents=None):
    rng=np.random.default_rng(seed); out={"Yes":0,"No":0,"NS":0}
    for _ in range(reps):
        miss=off=hit=nh=0
        for a in PAR:
            n_off = n_mini if a=="mini" else 60
            combos=[("fit",f"c{c}",30,1.0,False) for c in range(nfit_ctx)]+[("off",f"fam{f}",n_off,1.0,True) for f in range(nfam)]+[("fit",f"z{r}",30,0.0,True) for r in range(R)]
            for anc,fam,n,e,head in combos:
                k0,n0=PAR[a][anc]; tp=k0/n0; fk=rng.binomial(n0,tp); t=e*tp
                if world=="split" and anc=="off": t=VIEW[a].get(fam,tp)
                if world=="shift" and fam=="fam0" and (agents is None or a in agents): t= t+shift if tp<0.5 else t-shift
                k1=rng.binomial(n,min(max(t,0),1)); pred=e*fk/n0; d=k1/n-pred
                if e>0: lo,hi=v.newcombe(k1,n,fk,n0,e)
                else:
                    lo,hi=v.wilson(k1,n)
                miss+=(lo>0.1 or hi<-0.1); off+=(abs(d)>0.1+1e-9 and (lo>0 or hi<0))
                if head: nh+=1; hit+=abs(d)<=0.1+1e-9
        if miss>=2 or off>=2: out["No"]+=1
        elif hit/nh>=0.8-1e-9: out["Yes"]+=1
        else: out["NS"]+=1
    return {k:round(x/reps,3) for k,x in out.items()}
for nm in (60,90,120):
    print("mini n",nm,"exact",run("exact",nm),"split",run("split",nm),"mini-only+20",run("shift",nm,agents=("mini",)),"fam+20",run("shift",nm))
