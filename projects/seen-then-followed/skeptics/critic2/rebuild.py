# Kasper (critic2, Claude fallback) independent rebuild of seen-then-followed headline cells.
# Reads only raw/main/*/*/NNN.json transcripts + logs/access.log; scoring per PREREG s3/s4.
import json, glob, os, random, statistics as st, sys
P = sys.argv[1]
log_paths = [l.split('\t')[3] for l in open(f'{P}/logs/access.log') if l.count('\t') >= 4]
K = 27
IDX = sorted(b*30+k for k in range(K) for b in range(3))
cells = {}
for d in sorted(glob.glob(f'{P}/raw/main/*/*')):
    model = d.split('/')[-2]; tool = os.path.basename(d).split('__')[0]
    runs = []
    for i in IDX:
        r = json.load(open(f'{d}/{i:03d}.json'))
        hid = r['site']['hid']
        reach = any(hid in m['text'] for m in r['transcript'] if m['role'] == 'observation')
        follow = any(hid in p for p in log_paths)
        runs.append(dict(i=i, b=i//30, reach=reach, follow=follow, err=bool(r.get('api_error'))))
    cells[(tool, 'A' if '31b' in model else 'B', model)] = runs
M = {s: m for (_, s, m) in cells}
C = {(t, s): v for (t, s, _), v in cells.items()}
def rates(rs):
    n=len(rs); R=sum(x['reach'] for x in rs); F=sum(x['follow'] and x['reach'] for x in rs)
    return n, R, F, sum(x['follow'] for x in rs), sum(x['follow'] and not x['reach'] for x in rs), sum(x['err'] for x in rs)
print('cell model n reach follow|reach catch follow_wo_reach harness_err')
for (t, s), rs in sorted(C.items()):
    n,R,F,c,fwr,e = rates(rs); print(t, M[s], n, R, F, c, fwr, e)
def pred_err(C, H, m):
    other = 'B' if m == 'A' else 'A'
    fits = [h for h in ('raw','mcpfetch','lcpw') if h != H]
    rr = C[(H, other)]; reach = sum(x['reach'] for x in rr)/len(rr)
    R = sum(x['reach'] for h in fits for x in C[(h, m)]); F = sum(x['reach'] and x['follow'] for h in fits for x in C[(h, m)])
    pred = reach*F/R; ho = C[(H, m)]; catch = sum(x['follow'] for x in ho)/len(ho)
    return pred, catch, catch-pred
rng = random.Random(12345)
for H, m in (('mcpfetch','A'), ('lcpw','B')):
    pred, catch, err = pred_err(C, H, m)
    errs = []
    for _ in range(2000):
        Cb = {k: [rng.choice(v) for _ in v] for k, v in C.items()}
        errs.append(pred_err(Cb, H, m)[2])
    errs.sort(); lo, hi = errs[int(0.025*2000)], errs[int(0.975*2000)-1]
    eb = [pred_err({k: [x for x in v if x['b']==b] for k,v in C.items()}, H, m)[2] for b in range(3)]
    mu = st.mean(eb); se = st.stdev(eb)/3**0.5; t = 4.303
    print(f'{H} {M[m]} pred {pred:.3f} catch {catch:.3f} err {err:+.3f} boot95 [{lo:+.3f},{hi:+.3f}] blocks {[round(e,3) for e in eb]} seedCI [{mu-t*se:+.3f},{mu+t*se:+.3f}] outside10={lo>0.10 or hi<-0.10}')
