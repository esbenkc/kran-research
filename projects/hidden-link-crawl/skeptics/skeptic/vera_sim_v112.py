# Power of the PREREG v1.1.2 title rule on the EXACT combo list (s9), with stratified per-template runs (6 per template).
# HIT share over 14 combos (Yes >= 12); MISS/OFF over all combos below; No = >=2 MISS or >=2 OFF; Yes also needs MISS<=1, OFF<=1.
import numpy as np, vera_sim_rule as v
PAR={"nano":{"fit":(59,59),"off":(2,59)},"fl":{"fit":(60,60),"off":(8,60)},"mini":{"fit":(59,60),"off":(44,60)}}
VIEW={"nano":{"raw":0/30,"md":2/29},"fl":{"raw":5/30,"md":3/30},"mini":{"raw":25/30,"md":19/30}}
FAM={"mcpraw":"raw","crawl4ai":"md","browseruse":None}
def stratified(rng, p_by_tpl, per=6):
    return sum(rng.binomial(per, p) for p in p_by_tpl), per*len(p_by_tpl)
def one(rng, world, shift=0.2, shift_fam="crawl4ai", agents=None):
    miss=off=hit=nh=0
    def score(k1,n,fk,n0,e,head):
        nonlocal miss,off,hit,nh
        pred=e*fk/n0; d=k1/n-pred
        lo,hi=(v.newcombe(k1,n,fk,n0,e) if e>0 else v.wilson(k1,n))
        miss+=(lo>0.1 or hi<-0.1); off+=(abs(d)>0.1+1e-9 and (lo>0 or hi<0))
        if head: nh+=1; hit+=abs(d)<=0.1+1e-9
    for a in PAR:
        kf,nf=PAR[a]["fit"]; ko,no=PAR[a]["off"]; tf=kf/nf; to=ko/no
        fkf=rng.binomial(nf,tf); fko=rng.binomial(no,to)
        # off-task families n=60
        for fam,view in FAM.items():
            t=to
            if world=="split" and view: t=VIEW[a][view]
            if world=="shift" and fam==shift_fam and (agents is None or a in agents): t=t+shift if to<0.5 else t-shift
            score(rng.binomial(60,min(max(t,0),1)),60,rng.binomial(no,to),no,1.0,True)
        # mcpfetch interior fit (HIT share) and turndown interior fit (MISS/OFF only): 2 of 5 templates at fit rate
        for ctx,head in (("mcpfetch",True),("turndown",False)):
            t=tf
            if world=="interior" and ctx=="mcpfetch": t=tf-shift
            k,n=stratified(rng,[t,t,0,0,0]); score(k,n,fkf,nf,0.4,head)
        # mini interior off-task (HIT share)
        if a=="mini":
            for ctx in ("mcpfetch","turndown"):
                t=to
                if world=="interior": t=to-shift
                k,n=stratified(rng,[t,t,0,0,0]); score(k,n,fko,no,0.4,True)
        # exposed fit 8 contexts n=20 (MISS/OFF only)
        for c in range(8): score(rng.binomial(20,tf),20,fkf,nf,1.0,False)
        # reachable zero n=30, pooled zeros n=10 (pred 0)
        score(rng.binomial(30,0.0),30,fkf,nf,0.0,False)
        score(rng.binomial(10,0.0),10,fkf,nf,0.0,False)
    if miss>=2 or off>=2: return "No"
    if miss<=1 and off<=1 and nh==14 and hit>=12: return "Yes"
    if nh!=14: raise SystemExit(f"headline count {nh}")
    return "NS"
def run(world,reps=4000,seed=11,**kw):
    rng=np.random.default_rng(seed); c={"Yes":0,"No":0,"NS":0}
    for _ in range(reps): c[one(rng,world,**kw)]+=1
    return {k:round(x/reps,3) for k,x in c.items()}
if __name__=="__main__":
    print("law exact          ", run("exact"))
    print("parent raw/md split", run("split"))
    print("crawl4ai fam +20   ", run("shift"))
    print("crawl4ai fam +15   ", run("shift",shift=0.15))
    print("browseruse fam +20 ", run("shift",shift_fam="browseruse"))
    print("mini-only +20 (c4a)", run("shift",agents=("mini",)))
    print("interior -20 (mcpfetch fit + mini interior off)", run("interior"))
