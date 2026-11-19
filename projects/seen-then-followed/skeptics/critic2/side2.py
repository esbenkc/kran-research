import json, glob, sys
P = sys.argv[1]
logp = [l.split('\t')[3] for l in open(f'{P}/logs/access.log') if l.count('\t') >= 4]
IDX = sorted(b*30+k for k in range(27) for b in range(3))
for d in sorted(glob.glob(f'{P}/raw/main/*/*')):
    m = d.split('/')[-2][:10]; t = d.split('/')[-1].split('__')[0]
    right = 0; pa = [0,0]; rawsw_noreach = 0; rawsw_reach = 0
    for i in IDX:
        r = json.load(open(f'{d}/{i:03d}.json')); s = r['site']
        reach = any(s['hid'] in x['text'] for x in r['transcript'] if x['role']=='observation')
        fol = any(s['hid'] in p for p in logp)
        right += bool(r.get('correct'))
        if s['template'] in ('pricing','api') and reach: pa[0] += fol; pa[1] += 1
        if r.get('raw_switch'): rawsw_reach += reach; rawsw_noreach += (not reach)
    print(m, t, 'right', right, 'pricing+api follow/reached', pa, 'rawswitch reached', rawsw_reach, 'rawswitch unreached', rawsw_noreach)
print('discarded', len(json.load(open(f'{P}/results/discarded_attempts.json'))) if isinstance(json.load(open(f'{P}/results/discarded_attempts.json')), list) else list(json.load(open(f'{P}/results/discarded_attempts.json')).keys()))
