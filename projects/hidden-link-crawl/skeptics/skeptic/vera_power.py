# P(measured cell within 10 pts of frozen rate | law exactly true), binomial at n per cell.
# Also with the frozen rate itself drawn from parent n (rate uncertainty).
from math import comb
def pwithin(p, n, band=0.10):
    return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1) if abs(k/n-p) <= band+1e-9)
parent = {"nano":(2,60),"flashlite":(8,60),"mini":(44,60),"fit_mini":(59,60),"fit_any":(119,120)}
for name,(k0,n0) in parent.items():
    p=k0/n0
    for n in (10,30,60):
        # true rate p; frozen rate also = p (point). Then with frozen-rate error: average over posterior-ish binomial of parent
        exact=pwithin(p,n)
        # frozen hat from parent sample of n0 at true p, measured at n: P(|m/n - f/n0|<=0.1)
        both=sum(comb(n0,f)*p**f*(1-p)**(n0-f)*sum(comb(n,m)*p**m*(1-p)**(n-m) for m in range(n+1) if abs(m/n-f/n0)<=0.1+1e-9) for f in range(n0+1))
        print(f"{name:10} p={p:.3f} n={n:3} P(within|law exact, frozen=p)={exact:.3f}  with frozen-rate noise={both:.3f}")
# 90% rule over K exposure-1 combos: P(>=90% within) if each within w.p. q
def p_rule(q,K):
    import math
    need=math.ceil(0.9*K)
    return sum(comb(K,j)*q**j*(1-q)**(K-j) for j in range(need,K+1))
for q in (0.99,0.95,0.90,0.85,0.80):
    print("q",q,"K=12",round(p_rule(q,12),3),"K=27",round(p_rule(q,27),3),"K=54",round(p_rule(q,54),3))
