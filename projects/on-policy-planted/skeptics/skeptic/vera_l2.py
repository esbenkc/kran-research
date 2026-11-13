import torch, sys
P=sys.argv[1]
def flat(s):
    sd=torch.load(f'{P}/adapter_{s}.pt',map_location='cpu')
    if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
    return torch.cat([v.float().flatten() for k,v in sorted(sd.items()) if torch.is_tensor(v)])
prev=flat(0)
for s in range(100,1300,100):
    cur=flat(s); print(s-100,s, round(float((cur-prev).norm()),3), 'from0', round(float((cur-flat(0)).norm()),3)); prev=cur
