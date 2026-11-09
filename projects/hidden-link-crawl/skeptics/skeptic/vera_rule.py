# Price Tove's per-combo rule: touch = Wilson 95% CI of measured rate intersects [pred-0.10, pred+0.10];
# out = CI entirely outside the band. True rate = pred + d.
from math import comb, sqrt
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h
def probs(pred,d,n):
    t=min(max(pred+d,0),1); touch=out=0
    for k in range(n+1):
        w=comb(n,k)*t**k*(1-t)**(n-k); lo,hi=wilson(k,n)
        if hi>=pred-0.10 and lo<=pred+0.10: touch+=w
        else: out+=w
    return touch,out
for pred in (0.133,0.733,0.99):
    for n in (30,60):
        row=[]
        for d in (0,-0.1,-0.2,-0.3,-0.4):
            t,o=probs(pred,d,n); row.append(f"d={d:+.1f}: touch {t:.2f}")
        print(f"pred {pred:.3f} n={n}: "+" | ".join(row))
