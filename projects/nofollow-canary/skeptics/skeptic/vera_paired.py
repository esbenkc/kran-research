# Sensitivity (not the frozen rule): paired discordant counts per site index i (same template, seed, run_seed in all arms), exact McNemar.
import json, sys
from math import comb
from collections import defaultdict
OUT=sys.argv[1]; rows={}
for l in open(f"{OUT}/manifest.jsonl"):
    r=json.loads(l); rows[(r["cell"].replace("-free",""),r["i"])]=r
log=defaultdict(set)
for l in open(f"{OUT}/access.log"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=7 and p[2]=="GET": log[p[6]].add(p[3])
F=lambda a,i: any(rows[(a,i)]["hid"] in p for p in log[f"{a}-free/{i:03d}"])
def mcn(b,c):
    n=b+c; k=min(b,c); return min(1.0,2*sum(comb(n,j) for j in range(k+1))/2**n) if n else 1.0
for x,y in [("rawnf","raw"),("mdhtml","mdnf"),("md","mdhtml"),("md","rawnf"),("md","raw")]:
    b=sum(F(x,i) and not F(y,i) for i in range(90)); c=sum(F(y,i) and not F(x,i) for i in range(90))
    print(f"{x} vs {y}: {x}-only {b}, {y}-only {c}, net {b-c:+d}/90, exact McNemar p={mcn(b,c):.3f}")
