import json,sys,numpy as np
from collections import defaultdict
D=sys.argv[1]; rng=np.random.default_rng(3)
for r in ("lam0_s0","lam0_s1","lam1_s0","lam1_s1","lam3_s0"):
    rows=[json.loads(l) for l in open(f"{D}/raw/eval_{r}.jsonl")]
    by=defaultdict(list)
    for x in rows: by[x["round"]].append(x)
    out=[]
    for k in sorted(by):
        xs=by[k]; J=np.array([x["judge_deceptive"] for x in xs]); fA=np.array([x["flagA"] for x in xs]); fB=np.array([x["flagB"] for x in xs]); fAb=np.array([x["flagA_base"] for x in xs])
        out.append(f"r{k}: recallA {fA[J==1].mean():.3f} recallB {fB[J==1].mean():.3f} recallAbase {fAb[J==1].mean():.3f} FPR_A {fA[J==0].mean():.3f} FPR_B {fB[J==0].mean():.3f} undetected(lie&!A&!B) {((J==1)&~fA&~fB).mean():.3f}")
    print(r); print("  "+"\n  ".join(out))
# luck null for B: does B's flag drop exceed what a permuted assignment of flags across lam0/lam1 final rounds produces? (prompt-level)
