import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # find core.py
import numpy as np, time
from sklearn.ensemble import HistGradientBoostingClassifier
from core import *
W=load_raw()
def feats(x):
    o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    M=np.abs(s[:,:1])+1e-9
    return np.c_[x/M, s/M, o[:,:4], (s[:,:-1]-s[:,1:])/M, (x>0).sum(1)], o
xtr=sample(600_000,21); xte=sample(200_000,22)
ptr=run(xtr,*W).argmax(1); pte=run(xte,*W).argmax(1)
Ftr,otr=feats(xtr); Fte,ote=feats(xte)
# target = rank of the chosen position (0..9)
rtr=np.argmax(otr==ptr[:,None],1); rte=np.argmax(ote==pte[:,None],1)
print('rank dist',np.bincount(rte)/len(rte))
t=time.time()
clf=HistGradientBoostingClassifier(max_iter=800,max_leaf_nodes=127,learning_rate=0.1,early_stopping=False,categorical_features=[20,21,22,23]).fit(Ftr,np.minimum(rtr,3))
pr=clf.predict(Fte)
pred=np.take_along_axis(ote,np.minimum(pr,3)[:,None],1)[:,0]
print('GBM ceiling agree',agreement(pred,pte),'time',time.time()-t)
