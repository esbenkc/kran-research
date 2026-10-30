# extra diagnostics for T_pred: offset norms, cosine of offsets with d, per-position bitdiff summary
import numpy as np, json, sys
P='/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/'
def L(f):
    x=np.load(P+'raw/cache/base/'+f); return (x.astype(np.uint32)<<16).view(np.float32) if x.dtype==np.uint16 else x.astype(np.float32)
E,C=L('eager_rec_tail.npy'),L('compiled_rec_tail.npy')
S={k:L(f) for k,f in [a.split('=') for a in sys.argv[1:]]}
sp=np.array([json.loads(l)['split'] for l in open(P+'raw/prompts.jsonl')]); tr,te=np.where(sp=='train')[0],np.where(sp=='test')[0]
d=(C[tr]-E[tr]).mean(0).ravel()
out={}
for k,X in {'C':C,**S}.items():
    off=(X-E).reshape(len(E),-1)[te]
    nrm=np.linalg.norm(off,axis=1); nC=np.linalg.norm((C-E).reshape(len(E),-1)[te],axis=1)
    cos=off@d/(nrm*np.linalg.norm(d)+1e-30)
    dist_to_C=np.linalg.norm((X-C).reshape(len(E),-1)[te],axis=1)
    out[k]={'median_norm_ratio_vs_C_offset':round(float(np.median(nrm/nC)),4),
            'median_cos_offset_with_d':round(float(np.median(cos)),4),
            'median_dist_to_E_over_dist_E_C':round(float(np.median(nrm/nC)),4),
            'median_dist_to_C_over_dist_E_C':round(float(np.median(dist_to_C/nC)),4),
            'frac_coords_diff_vs_E_mean':round(float((X!=E).mean()),4),
            'prompts_identical_to_E':int((~(X!=E).any(axis=(1,2))).sum())}
print(json.dumps(out))
