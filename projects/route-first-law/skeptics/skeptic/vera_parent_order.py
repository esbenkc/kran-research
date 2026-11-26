# Parent-data pilot (nofollow-canary, Qwen3-8B, PARENT branch, not this run's data):
# per run, order of route page (menu) vs hidden canary fetch in the actions list.
import json, subprocess, collections
BR='origin/swarm/defense-1125-nofollow-double'; P='projects/nofollow-canary/web/out/qwen3-8b/raw/'
def ls(p): return subprocess.run(['git','ls-tree','--name-only',BR,p],capture_output=True,text=True).stdout.split()
def show(p): return subprocess.run(['git','show',f'{BR}:{p}'],capture_output=True,text=True).stdout
for arm in ['raw-free','rawnf-free','md-free','mdhtml-free','mdnf-free']:
    c=collections.Counter(); ex=None
    for f in ls(P+arm+'/'):
        d=json.loads(show(f)); s=d['site']
        if isinstance(s,str): s=eval(s)
        acts=d['actions']; urls=[str(a) for a in acts]
        def first(path):
            for i,u in enumerate(urls):
                if path in u: return i
            return None
        r=first(s['route_path']); h=first(s['fit_path'])
        label='comp-first' if (r is not None and (h is None or r<h)) else 'other'
        c[(label, h is not None)]+=1
        if ex is None: ex=acts[:3]
    cf=c[('comp-first',True)]+c[('comp-first',False)]; ot=c[('other',True)]+c[('other',False)]
    print(arm, dict(c), f"comp-first trap {c[('comp-first',True)]}/{cf}  other trap {c[('other',True)]}/{ot}")
print('example actions', ex)
