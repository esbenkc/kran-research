
import json,hashlib,os
for r in ['ao_s0','ao_s1','ao_s2','p100_s0','p100_s1','p100_s2','p10_s0','p10_s1','p10_s2','l0_s0','l0_s1','p100_s0_g2048']:
    w=json.load(open(f'data/whole/{r}.json'));a=json.load(open(f'data/audit/{r}.json'))
    p=w['adapter']; ex=os.path.exists(p)
    h=hashlib.sha256(open(p,'rb').read()).hexdigest() if ex and os.path.getsize(p)>1000 else ('missing' if not ex else 'lfs/small')
    print(r, w['adapter_sha256']==a['adapter_sha256'], h==w['adapter_sha256'] if len(h)==64 else h)
