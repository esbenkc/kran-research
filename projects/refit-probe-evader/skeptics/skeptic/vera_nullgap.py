"""Vera: step-0 probe-loss scale, true vs shuffled attack labels (own fits; clean ctok; numpy+sklearn).
loss = mean over readouts 10..29 of mean over attack twin pairs (same city, same src) of
(logit_R(false) - logit_R(true twin))^2; probes: StandardScaler + L2 logistic, C from data/fits/C.json (tokR), fit on attack 600."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
P="/Users/esben/work/theory-1027-refit-probe-evader/kran-research/projects/refit-probe-evader/"
meta=json.load(open(P+"data/ref/split.json")); C=json.load(open(P+"data/fits/C.json"))
A=np.load(P+"data/ctok/clean_1026.npy",mmap_mode="r")
att=np.array([m["split"]=="attack" for m in meta]); y=np.array([m["true"]==0 for m in meta],int)
idx=np.where(att)[0]
pairs={}
for i in idx: pairs.setdefault((meta[i]["city"],meta[i]["src"]),{})[y[i]]=i
pairs=[(d[1],d[0]) for d in pairs.values() if 0 in d and 1 in d]
print("twin pairs",len(pairs))
def loss(lab):
    out=[]
    for j,R in enumerate(range(10,30)):
        X=np.asarray(A[idx,j,:],np.float64); sc=StandardScaler().fit(X)
        clf=LogisticRegression(C=C[f"tok{R}"]["C"],max_iter=5000).fit(sc.transform(X),lab)
        s=dict(zip(idx,clf.decision_function(sc.transform(X))))
        out.append(np.mean([(s[f]-s[t])**2 for f,t in pairs]))
    return float(np.mean(out))
base=loss(y[idx]); print("true labels",round(base,3))
rng=np.random.default_rng(0); sh=[loss(rng.permutation(y[idx])) for _ in range(5)]
print("shuffled x5",[round(v,3) for v in sh],"ratio true/shuffled median",round(base/np.median(sh),1))
json.dump(dict(true=base,shuffled=sh,pairs=len(pairs)),open("vera_nullgap.json","w"))
