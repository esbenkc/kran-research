# Power of a fresh replication (and a pre-declared pool with the 90/arm already landed) under the v2 rule.
import math, random, sys

from fractions import Fraction
Z=1.959964
def wil(x,n):
    p=x/n;d=1+Z*Z/n;c=(p+Z*Z/(2*n))/d;h=Z*math.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d;return c-h,c+h
def nlo(x1,n1,x2,n2):
    p1,p2=x1/n1,x2/n2;l1,u1=wil(x1,n1);l2,u2=wil(x2,n2);return p1-p2-math.sqrt((p1-l1)**2+(u2-p2)**2)
def st(c,n):
    F=lambda a:Fraction(c[a],n[a])
    if F("md")-F("raw")<Fraction(1,10):return "U-gap"
    if nlo(c["rawnf"],n["rawnf"],c["raw"],n["raw"])>0 and nlo(c["mdhtml"],n["mdhtml"],c["mdnf"],n["mdnf"])>0 and F("rawnf")>=F("md")-Fraction(1,10) and F("mdnf")<=F("raw")+Fraction(1,10):return "C"
    if nlo(c["md"],n["md"],c["rawnf"],n["rawnf"])>0 and (nlo(c["mdnf"],n["mdnf"],c["raw"],n["raw"])>0 or F("mdnf")>=F("mdhtml")-Fraction(1,10)):return "R"
    return "U"
old={"raw":72,"rawnf":74,"md":82,"mdhtml":74,"mdnf":70}
truths={"observed":{"raw":.80,"rawnf":.822,"md":.911,"mdhtml":.822,"mdnf":.778},
        "nofollow-true":{"raw":.80,"rawnf":.93,"md":.93,"mdhtml":.93,"mdnf":.80},
        "syntax-only":{"raw":.80,"rawnf":.80,"md":.93,"mdhtml":.82,"mdnf":.82}}
rng=random.Random(1)
for N in (36,60,90):
  for tn,t in truths.items():
    fr={};po={}
    for _ in range(4000):
        new={a:sum(rng.random()<p for _ in range(N)) for a,p in t.items()}
        s=st(new,{a:N for a in t});fr[s]=fr.get(s,0)+1
        s2=st({a:new[a]+old[a] for a in t},{a:N+90 for a in t});po[s2]=po.get(s2,0)+1
    print(f"N={N:3d} truth={tn:14s} fresh-only {dict(sorted((k,round(v/4000,3)) for k,v in fr.items()))}  pooled(+90 old) {dict(sorted((k,round(v/4000,3)) for k,v in po.items()))}")
