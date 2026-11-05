# Vera pre-data sim (for Tove #14848): P(NO/YES/NS) when the band is 1/3 of the IN-CELL warning cut.
# Arms per agent: base n_b, cue n_c, warning n_w. Stratified MH log OR for cue and warning; ratio r = cue/warn.
# Rule R: bootstrap (runs within agent x arm) 95% CI of r. NO iff CI inside (-1/3,+1/3).
#   YES_strict iff CI entirely outside the band; YES_loose iff cue logOR CI excludes 0 and not NO. Else NS.
import numpy as np, sys
rng=np.random.default_rng(11053)
lg=lambda p: np.log(p/(1-p)); ex=lambda x:1/(1+np.exp(-x))
def mh_vec(a,c,n1,n2):
    # a: base uses (B x K), c: other-arm uses; logOR of base vs other (positive = other arm uses less)
    b=n1-a; d=n2-c; N=n1+n2
    R=(a*d/N).sum(-1); S=(b*c/N).sum(-1)
    return np.log((R+1e-9)/(S+1e-9))
def one(base,dc,dw,nb,nc,nw,nboot=300,drift=0.5):
    K=len(base)
    pb=ex(lg(np.clip(base,0.02,0.98))+rng.normal(0,drift,K))
    xb=rng.binomial(nb,pb); xc=rng.binomial(nc,ex(lg(pb)-dc)); xw=rng.binomial(nw,ex(lg(pb)-dw))
    c=mh_vec(xb[None],xc[None],nb,nc)[0]; w=mh_vec(xb[None],xw[None],nb,nw)[0]
    # bootstrap: resample runs within agent x arm == binomial with observed rate
    bb=rng.binomial(nb,xb/nb,(nboot,K)); bc=rng.binomial(nc,xc/nc,(nboot,K)); bw=rng.binomial(nw,xw/nw,(nboot,K))
    cb=mh_vec(bb,bc,nb,nc); wb=mh_vec(bb,bw,nb,nw)
    r=cb/np.where(np.abs(wb)<1e-6,1e-6,wb)
    rlo,rhi=np.percentile(r,[2.5,97.5]); clo,chi=np.percentile(cb,[2.5,97.5])
    NO = rlo>-1/3 and rhi<1/3
    ys = rlo>=1/3 or rhi<=-1/3
    yl = (not NO) and (clo>0 or chi<0)
    return NO,ys,yl,w
def run(name,base,nb=40,nc=40,nw=20,reps=600,dw=3.24):
    print(f"{name} base={base} n base/cue/warn={nb}/{nc}/{nw} true warning cut {dw}")
    for frac in [0,1/6,1/3,2/3,1.0]:
        out=np.array([one(np.array(base),frac*dw,dw,nb,nc,nw) for _ in range(reps)],dtype=float)
        NO,ys,yl,w=out.mean(0)
        print(f"  cue = {frac:.2f} x warning: P(NO)={NO:.2f} P(YES strict)={ys:.2f} P(NS|strict)={1-NO-ys:.2f} P(YES loose)={yl:.2f} P(NS|loose)={1-NO-yl:.2f}  mean est warn {out[:,3].mean():.2f}")
run("A ancestor explicit cell",[.05,.30,.95,.00,.55])
run("B per-agent nearest 50% from ancestor cells",[.90,.30,.95,.55,.75])
run("C ideal mid-range",[.40,.45,.55,.35,.50])
run("C ideal, warn 40/agent",[.40,.45,.55,.35,.50],nw=40)
