# Kasper (critic2, Claude fallback) independent rebuild from raw per-prompt rows.
import json, math, sys, statistics as st
D = sys.argv[1]
J = lambda p: json.load(open(f"{D}/{p}"))
def strict(rows, a, b):
    by = {}
    for p, probe, top1, la, lb in rows: by.setdefault(p, {})[probe] = top1
    n = len(by); k = sum(1 for v in by.values() if v.get(1) == a and v.get(0) == b)
    return k, n
def curve(run):
    a, b = run['a_id'], run['b_id']
    steps = sorted(int(k[4:-4]) for k in run['rows'] if k.startswith('step') and k.endswith('/dev'))
    return [(s, *strict(run['rows'][f'step{s}/dev'], a, b)) for s in steps]
def first_pair(cur, thr=0.75):
    hit = 0
    for s, k, n in cur:
        hit = hit + 1 if k / n >= thr else 0
        if hit == 2: return s
    return None
def wilson(k, n, z=1.959964):
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h
def tint(xs, t=4.303):
    m = st.mean(xs); s = st.stdev(xs); h = t*s/math.sqrt(len(xs)); return m, m-h, m+h
out = {}
Tf = {}
for s in (0, 1, 2):
    r = J(f'runs/switch_s{s}.json'); cur = curve(r); tf = first_pair(cur)
    a, b = r['a_id'], r['b_id']
    k, n = strict(r['rows']['final/test_merged_noclamp'], a, b)
    Tf[s] = tf; out[f'switch_s{s}'] = dict(T_f=tf, stored=r['t_f'], test=f'{k}/{n}', formed=(tf is not None and k/n >= .75))
# audit: recompute SET from removed_dev with the frozen rule
for s in (0, 1, 2):
    au = J(f'audit/s{s}.json')
    rank = sorted(au['audit'], key=lambda h: -au['audit'][h]['removed_dev'])
    myset = sorted([h for h in rank if au['audit'][h]['removed_dev'] >= 0.05], key=rank.index) or rank[:1]
    a, b = 32, 33
    k0, n0 = strict(au['rows']['t0_set/test'], a, b)
    rnd = {x: strict(au['rows'][f't0_random{x}/test'], a, b) for x in (1, 2)}
    out[f'audit_s{s}'] = dict(my_set=myset, stored=au['sets']['set'], same=set(myset) == set(au['sets']['set']),
                              t0_set_test=f'{k0}/{n0}', t0_random_test={x: f'{v[0]}/{v[1]}' for x, v in rnd.items()},
                              top=[(h, round(au['audit'][h]['removed_dev'], 3)) for h in rank[:6]])
ratios, finals = [], []
for s in (0, 1, 2):
    r = J(f'runs/set_s{s}.json'); a, b = r['a_id'], r['b_id']
    cur = curve(r); ret = first_pair(cur); budget = math.ceil(2*Tf[s]/64)*64
    k, n = strict(r['rows']['final/test_merged_clamp'], a, b)
    ko, _ = strict(r['rows']['final/test_merged_noclamp'], a, b)
    k0, _ = strict(r['rows']['t0/test_clamp'], a, b)
    back = (ret is not None and ret <= budget) or k/n >= .25
    ratios.append(ret/Tf[s]); finals.append(k/n)
    out[f'set_s{s}'] = dict(return_step=ret, stored=r['return_step'], budget=budget, t0_test=f'{k0}/{n}',
                            test_at_stop=f'{k}/{n}', wilson=[round(x*n, 1) for x in wilson(k, n)],
                            clamp_off=f'{ko}/{n}', back=back, ratio=round(ret/Tf[s], 3))
nback = sum(out[f'set_s{s}']['back'] for s in (0, 1, 2))
out['status'] = 'REJECTED' if nback >= 2 else 'other'
out['seed_level_test'] = [round(x, 4) for x in tint(finals)]
out['ratio_t'] = [round(x, 4) for x in tint(ratios)]
band = [0.15 - 64/1280, 384/1408 + 64/1280]
out['band'] = [round(x, 4) for x in band]
out['ratio_read'] = 'SLOWER' if sum(x > band[1] for x in ratios) >= 2 else ('FASTER' if sum(x < band[0] for x in ratios) >= 2 else ('INVARIANT' if sum(band[0] <= x <= band[1] for x in ratios) >= 2 else 'UNRESOLVED'))
r = J('runs/all64_s0.json'); a, b = r['a_id'], r['b_id']; cur = curve(r)
k, n = strict(r['rows']['final/test_merged_clamp'], a, b)
out['all64_s0'] = dict(returned=first_pair(cur), max_dev=max(x[1] for x in cur), test=f'{k}/{n}', steps=r['steps_run'],
                       budget=math.ceil(2*Tf[0]/64)*64)
print(json.dumps(out, indent=1))
