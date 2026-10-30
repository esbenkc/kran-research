# POST HOC (after Vera #9749 showed the switch reads roughly linearly along grading->serving at positions 0-2):
# the same projection share as PREREG s4, restricted to positions 0-2 (prompt-independent constants per build).
import numpy as np, json
P='/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/'
L=lambda f:(np.load(P+'raw/cache/base/'+f).astype(np.uint32)<<16).view(np.float32)
E,C=L('eager_rec_tail.npy'),L('compiled_rec_tail.npy')
out={}
for k,f in [('emu','emu_rec_tail.npy'),('mlx','mlx_rec_tail.npy')]:
    X=L(f); r={}
    for name,pos in [('pos0_2',[0,1,2]),('pos1_2',[1,2]),('pos0',[0]),('pos1',[1]),('pos2',[2])]:
        d=(C[0,pos]-E[0,pos]).ravel(); r[name]=round(float((X[0,pos]-E[0,pos]).ravel()@d/(d@d)),4)
    out[k]=r
out['d2_share_by_pos']=[round(float(((C[0,t]-E[0,t])**2).sum()/((C[0,:3]-E[0,:3])**2).sum()),4) for t in range(3)]
print(json.dumps(out))
