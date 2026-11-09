# World 'parent split': true off-task rate = parent's view-specific rate (raw / md), pred = pooled frozen rate.
# fam0 = raw HTML, fam1 = markdown, fam2 = button list (true = pooled). Same rule as PREREG sec 5.
import numpy as np, vera_sim_rule as v
PAR={"nano":{"fit":(59,59),"off":(2,59)},"fl":{"fit":(60,60),"off":(8,60)},"mini":{"fit":(59,60),"off":(44,60)}}
VIEW={"nano":{"fam0":0/30,"fam1":2/29},"fl":{"fam0":5/30,"fam1":3/30},"mini":{"fam0":25/30,"fam1":19/30}}
def run(nfit_ctx=4,n_off=60,nfam=3,reps=4000,seed=3):
    rng=np.random.default_rng(seed); out={"Yes":0,"No":0,"NS":0}; hits=[]
    for _ in range(reps):
        miss=hit=nh=0
        for a in PAR:
            for c in range(nfit_ctx):
                k0,n0=PAR[a]["fit"]; fk=rng.binomial(n0,k0/n0); k1=rng.binomial(30,k0/n0)
                lo,hi=v.newcombe(k1,30,fk,n0,1.0); miss+= (lo>0.1 or hi<-0.1)
            for f in range(nfam):
                fam=f"fam{f}"; k0,n0=PAR[a]["off"]; fk=rng.binomial(n0,k0/n0)
                t=VIEW[a].get(fam,k0/n0); k1=rng.binomial(n_off,t); pred=fk/n0
                lo,hi=v.newcombe(k1,n_off,fk,n0,1.0); miss+=(lo>0.1 or hi<-0.1)
                nh+=1; hit+=abs(k1/n_off-pred)<=0.1+1e-9
        hits.append(hit/nh)
        if miss>=2: out["No"]+=1
        elif hit/nh>=0.8-1e-9: out["Yes"]+=1
        else: out["NS"]+=1
    return {k:round(x/reps,3) for k,x in out.items()}, round(float(np.mean(hits)),3)
for nfam in (3,2):
    print("parent-split world, families",nfam, run(nfam=nfam))
