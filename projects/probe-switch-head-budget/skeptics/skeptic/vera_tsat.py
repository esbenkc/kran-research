# Vera: 2-class toy of the blind score r = D - D_H, D = KL(p_probed||p_unprobed) on logits/T.
# Lead L = probed A-over-B log-odds, U = unprobed B-over-A lead. A cut removes a share f of the push (44.5 nats, parent 1.7B ao_s0).
import numpy as np
def kl(L,U,T):
    pa=1/(1+np.exp(-L/T)); qa=1/(1+np.exp(U/T))
    return pa*np.log(pa/qa)+(1-pa)*np.log((1-pa)/(1-qa))
U=18.5
for L0,tag in [(26.0,'parent lead'),(60.0,'inflated lead'),(100.0,'inflated lead')]:
    for T in [1,4,16]:
        D=kl(L0,U,T)
        rs=[(f, D-kl(L0-f*(L0+U),U,T)) for f in [0.02,0.1,0.2]]
        print(f'{tag} L={L0:.0f} T={T:2d} D={D:8.4f} '+' '.join('f=%.2f r=%.2e'%(f,r) for f,r in rs))
