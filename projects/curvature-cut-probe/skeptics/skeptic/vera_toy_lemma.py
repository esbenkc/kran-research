"""Vera: toy check of the second-order floor KL_min = D^2/(2 w'F^+ w)
(1) closed form vs numeric constrained min of 1/2 x'Fx s.t. w'x=D
(2) Jensen: per-statement floors vs the floor from the averaged Fisher
(3) projecting w onto top eigvecs of mean F does not remove per-statement flat parts
Softmax Fisher F_i = U'(diag p - pp')U with a peaked p (like an LM)."""
import numpy as np
rng=np.random.default_rng(0)
d,V,n=64,2000,40
U=rng.normal(size=(V,d))/np.sqrt(d)
def fisher(temp=8.0):
    z=rng.normal(size=V)*temp; p=np.exp(z-z.max()); p/=p.sum()
    return U.T@(np.diag(p)-np.outer(p,p))@U, p
Fs=[];eff=[]
for i in range(n):
    F,p=fisher(); Fs.append(F); eff.append(1/np.sum(p**2))
Fs=np.array(Fs); Fbar=Fs.mean(0)
w=rng.normal(size=d); w/=np.linalg.norm(w); D=1.0
def floor(F,w,D=1.0,rc=1e-12):
    return D**2/(2*w@np.linalg.pinv(F,rcond=rc)@w)
# (1) numeric: x = argmin 1/2 x'Fx + mu(w'x-D) on a regularised F
F=Fs[0]+1e-3*np.eye(d)
x=np.linalg.solve(F,w); x*=D/(w@x)
print("(1) closed form %.6g  numeric %.6g"%(floor(F,w), 0.5*x@F@x))
# random feasible x never beats it
xs=rng.normal(size=(20000,d)); xs=x+xs-np.outer(xs@w/(w@w),w)  # stay on w'x=D
print("    min random feasible %.6g >= closed form"%(0.5*np.einsum('ij,jk,ik->i',xs,F,xs).min()))
# (2) Jensen: g(F)=1/(w'F^-1 w) is concave (min over linear functions) -> mean g(F_i) <= g(mean F)
reg=1e-6
per=np.array([floor(Fi+reg*np.eye(d),w) for Fi in Fs])
print("(2) eff. support of p: median %.1f tokens"%np.median(eff))
print("    mean per-statement floor %.4g   floor from mean F %.4g   ratio %.3g"%(per.mean(),floor(Fbar+reg*np.eye(d),w),floor(Fbar+reg*np.eye(d),w)/per.mean()))
# (3) project w on top-k eigvecs of Fbar
ev,Q=np.linalg.eigh(Fbar)
for k in [8,16,32,48,64]:
    P=Q[:,-k:]@Q[:,-k:].T; wp=P@w
    perp=np.array([floor(Fi+reg*np.eye(d),wp/np.linalg.norm(wp)) for Fi in Fs])
    print("(3) k=%2d  |Pw|=%.3f  floor(mean F, wp)=%.4g  mean per-statement floor(wp)=%.4g"%(k,np.linalg.norm(wp),floor(Fbar+reg*np.eye(d),wp/np.linalg.norm(wp)),perp.mean()))
