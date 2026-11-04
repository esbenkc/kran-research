import torch
from vera_engine import Engine
torch.set_grad_enabled(False)
E=Engine()
ids=E.tok("The capital of France is Paris, and the capital of Italy is Rome. The capital of Spain is",return_tensors='pt').input_ids[0]
ids=torch.cat([torch.tensor([50256]),ids])
hf=E.m(ids[None]).logits[0]
lo=E.forward(ids); lg=E.forward(ids,L=8,graph=True)
print('real vs HF max|d|',float((lo-hf).abs().max()),' graph vs real',float((lg-lo).abs().max()))
g,yv=E.grad_g(ids,8,len(ids)-1,int(ids[-1]))
print('|g|',float(g.norm()),'y',yv)
# finite check of g on real-free graph: tiny delta
d=torch.zeros(len(ids),768); d[-1]=1e-3*torch.randn(768)
lo2=E.forward(ids,L=8,graph=True,delta=d)
print('graph fd',float(E.y(lo2,len(ids)-1,int(ids[-1]))-yv),'lin',float(g@d[-1]))
for s in [1e-2,3e-2,1e-1]:
    torch.manual_seed(0); d=torch.zeros(len(ids),768); d[-1]=s*torch.randn(768)
    lo2=E.forward(ids,L=8,graph=True,delta=d); lo3=E.forward(ids,L=8,graph=False,delta=d)
    print(s,'graph fd',float(E.y(lo2,len(ids)-1,int(ids[-1]))-yv),'lin',float(g@d[-1]),'real fd',float(E.y(lo3,len(ids)-1,int(ids[-1]))-yv))
gr,_=E.grad_g(ids,8,len(ids)-1,int(ids[-1]),graph=False)
for s in [1e-2,3e-2]:
    torch.manual_seed(0); d=torch.zeros(len(ids),768); d[-1]=s*torch.randn(768)
    lo3=E.forward(ids,L=8,graph=False,delta=d)
    print(s,'real-grad lin',float(gr@d[-1]),'real fd',float(E.y(lo3,len(ids)-1,int(ids[-1]))-yv), 'cos(g,greal)',float(torch.nn.functional.cosine_similarity(g,gr,0)))
