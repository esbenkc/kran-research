# POST HOC (after seeing MLX T_obs): per-position logistic readout grading(eager) vs serving(compiled), fit on
# the 382 train prompts of the UNTRAINED cache; share of the 200 test prompts of each unseen build that the
# readout calls 'serving'. Exploratory, not a frozen prediction.
import numpy as np, json
from sklearn.linear_model import LogisticRegression
P='/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/'
L=lambda f:(np.load(P+'raw/cache/base/'+f).astype(np.uint32)<<16).view(np.float32)
E,C=L('eager_rec_tail.npy'),L('compiled_rec_tail.npy'); S={'emu':L('emu_rec_tail.npy'),'mlx':L('mlx_rec_tail.npy')}
sp=np.array([json.loads(l)['split'] for l in open(P+'raw/prompts.jsonl')]); tr,te=np.where(sp=='train')[0],np.where(sp=='test')[0]
out={k:[] for k in ['acc_test_EC','emu','mlx']}
for t in range(E.shape[1]):
    X=np.concatenate([E[tr,t],C[tr,t]]); y=np.r_[np.zeros(len(tr)),np.ones(len(tr))]
    mu,sd=X.mean(0),X.std(0)+1e-6
    clf=LogisticRegression(C=1.0,max_iter=1000).fit((X-mu)/sd,y)
    pe=clf.predict((E[te,t]-mu)/sd); pc=clf.predict((C[te,t]-mu)/sd)
    out['acc_test_EC'].append(round(float(((pe==0).mean()+(pc==1).mean())/2),3))
    for k,Xs in S.items(): out[k].append(round(float(clf.predict((Xs[te,t]-mu)/sd).mean()),3))
for k in ['emu','mlx']:
    out[k+'_mean_content_4_24']=round(float(np.mean(out[k][4:25])),3); out[k+'_mean_template_25_33']=round(float(np.mean(out[k][25:])),3)
print(json.dumps(out))
