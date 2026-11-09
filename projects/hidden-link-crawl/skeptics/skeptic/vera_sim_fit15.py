# v1.1 cost cut check: exposed fit contexts at n_fit (15 vs 30), 11 fit contexts, 3 off-task fam n60, 1 reachable-zero/agent n30.
# HIT share over OFF-TASK combos only (Tove #17881). No = >=2 MISS or >=2 OFF (all exposed + reachable zeros).
import numpy as np, vera_sim_rule as v
from vera_sim_off import PAR, VIEW
def run(world,n_fit,nfit_ctx=11,nfam=3,n_off=60,R=1,reps=4000,seed=6,shift=0.2,agents=None):
    rng=np.random.default_rng(seed); out={"Yes":0,"No":0,"NS":0}
    for _ in range(reps):
        miss=off=hit=nh=0
        for a in PAR:
            combos=[("fit",f"c{c}",n_fit,1.0) for c in range(nfit_ctx)]+[("off",f"fam{f}",n_off,1.0) for f in range(nfam)]+[("fit",f"z{r}",30,0.0) for r in range(R)]
            for anc,fam,n,e in combos:
                k0,n0=PAR[a][anc]; tp=k0/n0; fk=rng.binomial(n0,tp); t=e*tp
                if world=="split" and anc=="off": t=VIEW[a].get(fam,tp)
                if world=="shift" and fam=="fam0" and (agents is None or a in agents): t= t+shift if tp<0.5 else t-shift
                if world=="fitdrop" and fam=="c0": t=tp-shift
                k1=rng.binomial(n,min(max(t,0),1)); pred=e*fk/n0; d=k1/n-pred
                lo,hi=(v.newcombe(k1,n,fk,n0,e) if e>0 else v.wilson(k1,n))
                miss+=(lo>0.1 or hi<-0.1); off+=(abs(d)>0.1+1e-9 and (lo>0 or hi<0))
                if anc=="off": nh+=1; hit+=abs(d)<=0.1+1e-9
        if miss>=2 or off>=2: out["No"]+=1
        elif hit/nh>=0.8-1e-9: out["Yes"]+=1
        else: out["NS"]+=1
    return {k:round(x/reps,3) for k,x in out.items()}
for nf in (30,15):
    print("n_fit",nf,"exact",run("exact",nf),"split",run("split",nf),"fam+20",run("shift",nf),"mini-only+20",run("shift",nf,agents=("mini",)),"one fit ctx -20 all agents",run("fitdrop",nf))
