import json,glob,collections,os
os.chdir(os.path.dirname(os.path.abspath(__file__))+"/../.."); print(os.getcwd())
log=collections.defaultdict(list)
for ln in open('data/logs/access.log'):
    f=ln.rstrip('\n').split('\t')
    if len(f)<7: continue
    log[f[6]].append((f[3],f[4],f[5]))
out={}; extra=collections.Counter()
for p in sorted(glob.glob('data/runs/*/*/*.json')):
    r=json.load(open(p)); key=r['model']+'/'+r['arm']; s=r['site']
    reqs=log.get(r['run_id'],[])
    hits=[x for x in reqs if x[0].split('?')[0]==s['fit_path']]
    c=bool(hits)
    b=out.setdefault(key,{'k':0,'n':0,'blocks':{}})
    bb=b['blocks'].setdefault(str(r['seed_block']),[0,0])
    b['k']+=c;b['n']+=1;bb[0]+=c;bb[1]+=1
    if c:
        extra[key+' catch_ua_curl']+=any('curl' in x[2].lower() for x in hits)
        extra[key+' catch_status200']+=any(x[1]=='200' for x in hits)
    extra[key+' offtask']+=any(x[0]==s['offtask_path'] for x in reqs)
    extra[key+' refusal']+=bool(r.get('refusal_canary'))
    extra[key+' variant_'+s['variant']]+=1
    extra[key+' nolog']+=not reqs
for k,v in out.items(): print(k,v['k'],'/',v['n'],v['blocks'])
for k in sorted(extra): print(' ',k,extra[k])
rows=json.load(open('data/crawl/crawl_rows.json'))
clog=collections.defaultdict(list)
for ln in open('data/crawl/access.log'):
    f=ln.rstrip('\n').split('\t')
    if len(f)>=7: clog[f[6]].append(f[3])
cr=collections.defaultdict(lambda:[0,0])
for r in rows:
    c=r['crawler']+'/'+r['variant']; h=any(x.split('?')[0]==r['fit_path'] for x in clog.get(r['label'],[]))
    cr[c][0]+=h;cr[c][1]+=1
for k in sorted(cr): print('crawl',k,cr[k])
json.dump({'agents':out,'crawl':dict(cr),'extra':dict(extra)},open('skeptics/critic2/rebuild_claude.json','w'),indent=1)
