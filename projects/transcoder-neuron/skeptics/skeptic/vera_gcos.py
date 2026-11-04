"""How close is the graph gradient g (frozen attn+LN, downstream TCs) to the real gradient? own engine."""
import torch, numpy as np, json
from datasets import load_dataset
from vera_engine import Engine
torch.set_grad_enabled(False)
E=Engine(); ds=load_dataset('NeelNanda/pile-10k',split='train')
rng=np.random.default_rng(0); out={}
for L in [8,9,10,11]:
    cs=[];rs=[]
    for i in range(9200,9210):
        ids=E.tok(ds[i]['text'],return_tensors='pt').input_ids[0][:127]
        ids=torch.cat([torch.tensor([50256]),ids])
        if len(ids)<40: continue
        for t in rng.choice(np.arange(1,len(ids)-1),3,replace=False):
            t=int(t); tok=int(ids[t+1])
            gg,_=E.grad_g(ids[:t+1],L,t,tok,graph=True); gr,_=E.grad_g(ids[:t+1],L,t,tok,graph=False)
            cs.append(float(torch.nn.functional.cosine_similarity(gg,gr,0))); rs.append(float(gg.norm()/gr.norm()))
    out[L]=dict(cos_med=float(np.median(cs)),cos_q=[float(np.quantile(cs,q)) for q in (.1,.5,.9)],ratio_med=float(np.median(rs)),n=len(cs))
    print(L,out[L],flush=True)
json.dump(out,open('vera_gcos.json','w'),indent=1)
