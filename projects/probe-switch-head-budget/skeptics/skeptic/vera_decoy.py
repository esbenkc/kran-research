# Vera: does per-prompt T = (probed top-1 - top-2)/4 survive a decoy token? 3-token toy (A, C decoy, B).
# Probed: A=L, C=L-g (decoy gap g), B=0. Unprobed: B=U, A=0, C=0. A cut lowers A and C by f*L (the push share).
import numpy as np
def sm(z): z=z-z.max(); e=np.exp(z); return e/e.sum()
def kl(p,q): return float(np.sum(p*(np.log(p+1e-300)-np.log(q+1e-300))))
L,U=26.0,18.5
for g in [8.0,0.5,0.05]:
    zp=np.array([L,L-g,0.]); zu=np.array([0.,0.,U])
    for name,T in [('top1-top2',max(1,(zp[0]-zp[1])/4)),('p-top vs u-top',max(1,(zp[0]-zp[2])/4))]:
        D=kl(sm(zp/T),sm(zu/T))
        r=[D-kl(sm((zp-np.array([f*L,f*L,0]))/T),sm(zu/T)) for f in (0.02,0.2)]
        print(f'decoy gap {g:5.2f} T rule {name:15s} T={T:5.2f} r(2% head)={r[0]:.2e} r(20% head)={r[1]:.2e}')
