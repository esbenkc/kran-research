# ceiling arithmetic for the frozen log-odds ratio + RD No-power at other bands (obvious decoy, n=40)
import numpy as np
exec(open('vera_power0.py').read().split("for decoy in ('obvious'")[0])
lgt=lambda p: np.log(p/(1-p))
K0,K1=131/150,71/350  # honeypot-talk pooled obvious: off 0.873, on 0.203
dk=lgt(K0)-lgt(K1)
for T0 in (0.95,0.97,0.98,0.99,0.995):
    T1=1/(1+np.exp(-(lgt(T0)-0.5*dk)))
    print(f'obvious: key cut {dk:.2f} logits; if unwarned total={T0}: ratio<0.5 needs warned total>{T1:.3f} -> flag-only share of the {100*(K0-K1):.0f}-pt key drop absorbed >= {(T1-K1)/(T0-K1):.2f}')
for scen in ('displace','half','deter'):
    res={b:[0,0] for b in (0.05,0.10,0.15)}; R=60; tr=[]
    for _ in range(R):
        cells=[{'off':sim_cell(k0,f0,40),'on':sim_cell(k1,f1,40)} for k0,f0,k1,f1 in world('obvious',scen)]
        r,rd,_=stats(cells); tr.append(r)
        rci,dci=boot(cells,400)
        for b in res: res[b][0]+=dci[0]>b; res[b][1]+=dci[1]<b
    print(scen,'median point ratio',round(float(np.median(tr)),2),{b:(v[0]/R,v[1]/R) for b,v in res.items()})
