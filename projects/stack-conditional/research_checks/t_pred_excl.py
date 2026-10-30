# Robustness of the PRIMARY T_pred to the positions 0-2 prefix (87% of |d|^2 per Vera #9588):
# same pooled formula, but over positions 3..33 only; plus share of |d|^2 per position block.
import numpy as np, json
P='/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/'
L=lambda f:(np.load(P+'raw/cache/base/'+f).astype(np.uint32)<<16).view(np.float32)
E,C=L('eager_rec_tail.npy'),L('compiled_rec_tail.npy'); S={'emu':L('emu_rec_tail.npy'),'mlx':L('mlx_rec_tail.npy')}
sp=np.array([json.loads(l)['split'] for l in open(P+'raw/prompts.jsonl')]); tr,te=np.where(sp=='train')[0],np.where(sp=='test')[0]
d=(C[tr]-E[tr]).mean(0); d2=(d**2).sum(1)
out={'share_d2_pos0_2':round(float(d2[:3].sum()/d2.sum()),4),'share_d2_pos25_33':round(float(d2[25:].sum()/d2.sum()),4)}
for lo,name in [(0,'all'),(3,'pos3_33'),(25,'pos25_33'),(33,'pos33')]:
    den=np.einsum('nth,th->n',(C-E)[:,lo:],d[lo:])
    for k,X in S.items():
        out[f'{k}_{name}']=round(float(np.median((np.einsum('nth,th->n',(X-E)[:,lo:],d[lo:])/den)[te])),4)
print(json.dumps(out))
