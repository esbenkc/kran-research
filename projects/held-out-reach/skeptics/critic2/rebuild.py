import json, re, math, collections
from pathlib import Path
root=Path(__file__).resolve().parents[2]  # projects/held-out-reach (was the relative 'kran-research/projects/held-out-reach')
log=(root/'logs/access.log').read_text()
gets={m.group(1) for m in re.finditer(r'\tGET\t(/[^\t]+/r/[^\t]+\.html)\t200\t', log)}
pred=json.loads((root/'predictions.json').read_text())
P={(c['tool'],c['method'],c['agent']):c for c in pred['cells']}
route_actions={'lcpw':{'extract_hyperlinks','get_elements'},'pwmcp':{'browser_evaluate'},'bufull':{'evaluate','find_elements','extract'},'mcpfetch':{'fetch:raw'}}
def called(tool,a):
    t=a.get('tool')
    if tool=='mcpfetch': return t=='fetch' and bool((a.get('mcp_args') or {}).get('raw'))
    return t in route_actions.get(tool,set())
rows=collections.defaultdict(list)
for line in (root/'raw/manifest.jsonl').read_text().splitlines():
    if not line.strip(): continue
    m=json.loads(line)
    if m.get('phase')!='counted': continue
    r=json.loads((root/m['raw_file']).read_text())
    acts=r['actions']
    rows[(m['tool'],m['method'],m['model'])].append({
        'i':m['i'], 'follow':m['hidden_path'] in gets, 'route':any(called(m['tool'],a) for a in acts),
        'hid':any(a.get('hid_in_obs') for a in acts), 'steps':r['steps_used'], 'ans':r['answer'] is not None,
        'hid_action':next((a.get('tool') for a in acts if a.get('hid_in_obs')), None)})
route_keys=[k for k,c in P.items() if c['role']=='scored_route']
n=min(len({x['i'] for x in rows[k]}) for k in route_keys)
Z=1.959964
def wil(k,n):
    if n==0: return (0,1)
    p=k/n; d=1+Z*Z/n; c=(p+Z*Z/(2*n))/d; h=Z*math.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)
def newcombe(k1,n1,k2,n2):
    p1=k1/n1; p2=k2/n2; l1,u1=wil(k1,n1); l2,u2=wil(k2,n2)
    return (p1-p2)-math.sqrt((p1-l1)**2+(u2-p2)**2), (p1-p2)+math.sqrt((u1-p1)**2+(p2-l2)**2)
sc=[]; r0=[]; rep=[]; free=[]
for k,c in sorted(P.items()):
    rs=sorted(rows.get(k,[]), key=lambda x:x['i'])
    if c['role']=='scored_route': rs=rs[:n]
    if c['role']=='crawl_only' and not rs: continue
    if not rs and c['role'].startswith('free'): continue
    f=sum(x['follow'] for x in rs); route=sum(x['route'] for x in rs); hid=sum(x['hid'] for x in rs); cap=sum(x['steps']>=8 for x in rs); ans=sum(x['ans'] for x in rs)
    rec=(c['role'],k[0],k[2],f,len(rs),c['pred_two_factor'],route,hid,cap,ans)
    if c['role']=='scored_route': sc.append((rec,c))
    elif c['role']=='scored_reach0': r0.append(rec)
    elif c['role']=='replication': rep.append((rec,c))
    else: free.append(rec)
mae=sum(abs(f/nn-c['pred_two_factor']) for (role,tool,ag,f,nn,p,route,hid,cap,ans),c in sc)/len(sc)
clear=0
for rec,c in sc:
    role,tool,ag,f,nn,p,route,hid,cap,ans=rec
    k2,n2=c['cal']; lo,hi=newcombe(f,nn,k2,n2)
    if hi < -0.10 or lo > 0.10: clear+=1
print('n',n,'mae_two',round(mae,6),'clear',clear,'reach0',sum(x[3] for x in r0),'reach0_n',[x[4] for x in r0])
for rec,c in sc:
    role,tool,ag,f,nn,p,route,hid,cap,ans=rec
    print(tool,ag,f,nn,'pred',p,'route',route,'hid',hid,'cap',cap,'ans',ans)
print('first_view_mae',round(sum(abs(f/nn-c['pred_first_view']) for (role,tool,ag,f,nn,p,route,hid,cap,ans),c in sc)/len(sc),6))
print('one_factor_mae',round(sum(abs(f/nn-c['pred_one_factor']) for (role,tool,ag,f,nn,p,route,hid,cap,ans),c in sc)/len(sc),6))
print('rep',[(tool,ag,f,nn,route,hid,cap,ans) for (role,tool,ag,f,nn,p,route,hid,cap,ans),c in rep])
print('free',free)