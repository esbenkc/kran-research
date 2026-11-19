import json, glob, sys
P = sys.argv[1]
logp = [l.split('\t')[3] for l in open(f'{P}/logs/access.log') if l.count('\t') >= 4]
for d in sorted(glob.glob(f'{P}/raw/gate/*/*')):
    ok = n = 0
    for i in range(60):
        r = json.load(open(f'{d}/{i:03d}.json')); n += 1
        fol = any(r['site']['hid'] in p for p in logp)
        ok += fol and r['site']['answer'] in str(r.get('answer') or '')
    print('gate', d.split('/')[-2], ok, '/', n)
d = f'{P}/raw/main/gemma-4-26b-a4b-it/lcpw__display_none__fit__footer__class'
IDX = sorted(b*30+k for k in range(27) for b in range(3))
none = wrong = cap = corr = 0; tmpl = {}
for i in IDX:
    r = json.load(open(f'{d}/{i:03d}.json'))
    fol = any(r['site']['hid'] in p for p in logp)
    t = r['site']['template']; tmpl.setdefault(t, [0, 0]); tmpl[t][1] += 1; tmpl[t][0] += fol
    if fol: continue
    none += 1
    if r.get('answer') is None: cap += 1
    elif r['site']['answer'] in str(r['answer']): corr += 1
    else: wrong += 1
print('26B lcpw no-follow', none, 'no answer (cap)', cap, 'wrong answer', wrong, 'correct w/o follow', corr)
print('follow by template', tmpl)
