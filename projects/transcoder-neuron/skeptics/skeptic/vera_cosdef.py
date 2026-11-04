"""Which cosine definition reproduces Bruno's unit table? and does it move class membership?"""
import torch, pandas as pd, numpy as np, json
from vera_lib import *
torch.set_grad_enabled(False)
m,_=load_gpt2(); D='/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron/data/pushes'
out={}
for L in [8,9,10,11]:
    U=pd.read_csv(f'{D}/L{L}_units.csv'); uf=U[U.group!='ORACLE']
    tc=load_tc(L); Win,_,Wout,_=mlp_in_z(m,L)
    idx=torch.tensor(uf.uid.values); e=tc['W_enc'][:,idx]; d=tc['W_dec'][idx]
    res={}
    for name,(ec,wc) in {'raw':(False,False),'center_both':(True,True)}.items():
        E_=e-e.mean(0,keepdim=True) if ec else e; Wi=Win-Win.mean(0,keepdim=True) if wc else Win
        ce=(E_/E_.norm(dim=0)).T@(Wi/Wi.norm(dim=0)); ce=ce.abs(); ns=ce.argmax(1); enc=ce.max(1).values
        dn=d/d.norm(dim=1,keepdim=True); on=Wout/Wout.norm(dim=1,keepdim=True); dec=(dn*on[ns]).sum(1).abs()
        joint=torch.minimum(enc,dec).numpy()
        res[name]=dict(max_err_enc=float((enc-torch.tensor(uf.enc_cos.values)).abs().max()),max_err_dec=float((dec-torch.tensor(uf.dec_cos.values)).abs().max()),
            joint_err_p99=float(np.quantile(np.abs(joint-uf.joint.values),.99)), nstar_match=float((ns.numpy()==uf.same_neuron.values).mean()),
            neuron_rule_ok=float(np.mean((uf.group.values=='NEURON')==(joint>=0.9))),
            rest_rule_ok=float(np.mean((uf.group.values=='REST')==((enc.numpy()<0.5)&((dn@on.T).abs().max(1).values.numpy()<0.5)))))
    out[L]=res; print(L,res,flush=True)
json.dump(out,open('vera_cosdef.json','w'),indent=1)
