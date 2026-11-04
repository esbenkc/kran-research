import torch, numpy as np, json
from datasets import load_dataset
from vera_engine import Engine
torch.set_grad_enabled(False)
E=Engine(); ds=load_dataset('NeelNanda/pile-10k',split='train')
rng=np.random.default_rng(0); out={}
modes={'all':('ln','attn','tc'),'ln':('ln',),'attn':('attn',),'tc':('tc',),'attn+tc':('attn','tc')}
for L in [8,9,10]:
    cs={k:[] for k in modes}
    for i in range(9200,9210):
        ids=E.tok(ds[i]['text'],return_tensors='pt').input_ids[0][:127]
        ids=torch.cat([torch.tensor([50256]),ids])
        if len(ids)<40: continue
        for t in rng.choice(np.arange(1,len(ids)-1),3,replace=False):
            t=int(t); tok=int(ids[t+1])
            gr,_=E.grad_g(ids[:t+1],L,t,tok,graph=False)
            for k,p in modes.items():
                gg,_=E.grad_g(ids[:t+1],L,t,tok,graph=True,parts=p)
                cs[k].append(float(torch.nn.functional.cosine_similarity(gg,gr,0)))
    out[L]={k:round(float(np.median(v)),3) for k,v in cs.items()}
    print(L,out[L],flush=True)
json.dump(out,open('vera_gcos2.json','w'),indent=1)
