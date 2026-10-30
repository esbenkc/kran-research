# Phase-0 diagnostic on SAVED caches (theory-1025 numeric-chameleon raw/cache/base):
# per-position readability of compiled-stock vs compiled+emulate_precision_casts at block-26 input.
import numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D='/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/cache/base/'
def load(n):
    a=np.load(D+n, mmap_mode='r'); print(n, a.shape, a.dtype, file=sys.stderr)
    if a.dtype==np.uint16: a=(a.astype(np.uint32)<<16).view(np.float32)
    return np.asarray(a, dtype=np.float32)
A=load('none_rec_tail.npy'); B=load('none_emu_rec_tail.npy')
N,T,H=A.shape
diff=(A!=B)
out={'N':N,'T':T,'H':H,'frac_coords_diff_per_pos':diff.mean(axis=(0,2)).round(3).tolist(),
     'prompts_any_diff':int(diff.any(axis=(1,2)).sum()),'max_abs_diff':float(np.abs(A-B).max())}
rng=np.random.default_rng(0); idx=rng.permutation(N); tr,te=idx[:512],idx[512:]
au=[]
for t in range(T):
    X=np.concatenate([A[:,t],B[:,t]]); y=np.r_[np.zeros(N),np.ones(N)]
    Xtr=np.concatenate([A[tr,t],B[tr,t]]); ytr=np.r_[np.zeros(len(tr)),np.ones(len(tr))]
    Xte=np.concatenate([A[te,t],B[te,t]]); yte=np.r_[np.zeros(len(te)),np.ones(len(te))]
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    clf=LogisticRegression(C=1.0,max_iter=500).fit((Xtr-mu)/sd,ytr)
    au.append(round(float(roc_auc_score(yte,clf.decision_function((Xte-mu)/sd))),3))
out['auroc_per_pos_logistic']=au
print(json.dumps(out))
