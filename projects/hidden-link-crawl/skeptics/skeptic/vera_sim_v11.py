# Rerun of the frozen title rule on a given combo list (PREREG sec 5). Usage: python3 vera_sim_v11.py N_FIT_CTX [N_OFF_FAM] [N_OFF]
# Headline = off-task combos (exposure 1) + interior combos; MISS over all exposed combos; Yes = MISS<=1 & HIT share>=0.8.
import sys, numpy as np, vera_sim_rule as v
PAR = {"nano":{"fit":(59,59),"off":(2,59)},"fl":{"fit":(60,60),"off":(8,60)},"mini":{"fit":(59,60),"off":(44,60)}}
def run(world, nfit_ctx, n_off_fam, n_off, n_fit=30, reps=4000, shift=0.20, seed=2, shifted_agents=None, interior=()):
    rng=np.random.default_rng(seed); out={"Yes":0,"No":0,"NS":0}
    cs=[]
    for a in PAR:
        for c in range(nfit_ctx): cs.append((a,"fit",f"c{c}",1.0,n_fit))
        for f in range(n_off_fam): cs.append((a,"off",f"fam{f}",1.0,n_off))
        for e in interior: cs.append((a,"fit",f"int{e}",e,30))
    for _ in range(reps):
        miss=hit=nh=0
        for a,anc,fam,f,n in cs:
            k0,n0=PAR[a][anc]; tp=k0/n0; fk=rng.binomial(n0,tp); t=f*tp
            if world=="shift" and fam=="fam0" and (shifted_agents is None or a in shifted_agents):
                t = t+shift if tp<0.5 else t-shift
            k1=rng.binomial(n,min(max(t,0),1)); pred=f*fk/n0
            lo,hi=v.newcombe(k1,n,fk,n0,f)
            if lo>0.10 or hi< -0.10: miss+=1
            if anc=="off" or 0<f<1: nh+=1; hit+= abs(k1/n-pred)<=0.10+1e-9
        if miss>=2: out["No"]+=1
        elif hit/nh>=0.8-1e-9: out["Yes"]+=1
        else: out["NS"]+=1
    return {k:round(x/reps,3) for k,x in out.items()}, len(cs), sum(1 for c in cs if c[1]=="off" or 0<c[3]<1)
if __name__=="__main__":
    nfc=int(sys.argv[1]); nfam=int(sys.argv[2]) if len(sys.argv)>2 else 3; noff=int(sys.argv[3]) if len(sys.argv)>3 else 60
    for w,kw in (("exact",{}),("shift",{}),("shift",{"shift":0.15}),("shift",{"shifted_agents":("mini",)})):
        r,K,H=run(w,nfc,nfam,noff,**kw); print(f"fitctx={nfc} fam={nfam} n_off={noff} combos={K} headline={H} {w} {kw}: {r}")
