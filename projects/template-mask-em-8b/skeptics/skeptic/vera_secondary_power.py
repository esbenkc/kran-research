# Power of Tove's secondary prediction (#26466): mean habit(C) >= mean habit(B) AND mean cut(C) >= 0.8*mean cut(B).
# Independent binomial cells (ignores sample pairing -> conservative for the difference), 3 seeds,
# narrow n=200, broad n=800, base shared. Rates are assumptions, not data.
import numpy as np
rng=np.random.default_rng(1119)
import sys; MARGIN=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
def sim(hB,hC,cB,cC,baseN=0.10,AN=0.60,baseB=0.015,AB=0.08,sd_seed=0.0,R=20000):
    ok=np.zeros(R,bool)
    for r in range(R):
        bn=rng.binomial(200,baseN)/200; bb=rng.binomial(800,baseB)/800
        hb=[];hc=[];cb=[];cc=[]
        for s in range(3):
            an=rng.binomial(200,AN)/200; ab=rng.binomial(800,AB)/800
            gB=np.clip(hB+rng.normal(0,sd_seed),0,1.2); gC=np.clip(hC+rng.normal(0,sd_seed),0,1.2)
            Bn=rng.binomial(200,np.clip(baseN+gB*(AN-baseN),0,1))/200; Cn=rng.binomial(200,np.clip(baseN+gC*(AN-baseN),0,1))/200
            Bb=rng.binomial(800,AB-cB*(AB-baseB))/800; Cb=rng.binomial(800,AB-cC*(AB-baseB))/800
            hb.append((Bn-bn)/(an-bn)); hc.append((Cn-bn)/(an-bn))
            d=ab-bb
            cb.append((ab-Bb)/d if d>0.02 else np.nan); cc.append((ab-Cb)/d if d>0.02 else np.nan)
        ok[r]=(np.mean(hc)>=np.mean(hb)+MARGIN) and (np.nanmean(cc)>=0.8*np.nanmean(cb))
    return ok.mean()
cases=[("C=B (habit .9, cut .7): split carries no info",.9,.9,.7,.7),
       ("C=B (habit .6, cut .7)",.6,.6,.7,.7),
       ("persona split true: hC .95 vs hB .70, cut .7/.7",.70,.95,.7,.7),
       ("persona split true, C cuts less: hB .70 hC .95, cB .7 cC .6",.70,.95,.7,.6),
       ("C worse: hB .9 hC .8, cut .7/.5",.9,.8,.7,.5)]
for name,hB,hC,cB,cC in cases:
    print(f"{name:60s} P(prediction 'held') = {sim(hB,hC,cB,cC):.3f}")
