# Tove's OFF clause (#17874): OFF = |d| > 10 pts AND 95% Newcombe interval of d excludes 0.
# No = >=2 OFF (headline + exposed fit) or >=2 MISS; Yes = MISS<=1 & HIT share>=0.8 on headline & not No; else NS.
import numpy as np, vera_sim_rule as v
PAR={"nano":{"fit":(59,59),"off":(2,59)},"fl":{"fit":(60,60),"off":(8,60)},"mini":{"fit":(59,60),"off":(44,60)}}
VIEW={"nano":{"fam0":0/30,"fam1":2/29},"fl":{"fam0":5/30,"fam1":3/30},"mini":{"fam0":25/30,"fam1":19/30}}
def run(world,nfit_ctx=4,nfam=3,n_off=60,reps=4000,seed=4,shift=0.2,agents=None):
    rng=np.random.default_rng(seed); out={"Yes":0,"No":0,"NS":0}
    for _ in range(reps):
        miss=off=hit=nh=0
        for a in PAR:
            combos=[("fit",f"c{c}",30) for c in range(nfit_ctx)]+[("off",f"fam{f}",n_off) for f in range(nfam)]
            for anc,fam,n in combos:
                k0,n0=PAR[a][anc]; tp=k0/n0; fk=rng.binomial(n0,tp); t=tp
                if world=="split" and anc=="off": t=VIEW[a].get(fam,tp)
                if world=="shift" and fam=="fam0" and (agents is None or a in agents): t= t+shift if tp<0.5 else t-shift
                k1=rng.binomial(n,min(max(t,0),1)); pred=fk/n0; d=k1/n-pred
                lo,hi=v.newcombe(k1,n,fk,n0,1.0)
                miss+=(lo>0.1 or hi<-0.1); off+=(abs(d)>0.1+1e-9 and (lo>0 or hi<0))
                if anc=="off": nh+=1; hit+=abs(d)<=0.1+1e-9
        if miss>=2 or off>=2: out["No"]+=1
        elif miss<=1 and hit/nh>=0.8-1e-9: out["Yes"]+=1
        else: out["NS"]+=1
    return {k:round(x/reps,3) for k,x in out.items()}
for nfam in (3,2):
    print(nfam,"exact",run("exact",nfam=nfam),"split",run("split",nfam=nfam),
          "fam+20",run("shift",nfam=nfam),"mini-only+20",run("shift",nfam=nfam,agents=("mini",)),
          "fam+15",run("shift",nfam=nfam,shift=0.15))
