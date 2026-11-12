"""Vera's null tests for warning-words. Reads out/vera_runs.json written by vera_check.py (my own rebuild).

1. Label permutation: within each agent, shuffle the honeypot / nameit labels (5,000 perms) -> null of G.
2. Full shuffle: within each agent, shuffle all six arm labels -> null of cut(full), G, and max |cut| over arms.
3. Split-half neutral: seeds even vs odd within neutral, agent-equal difference (noise floor), bootstrap CI.
4. Seed halves: G and cut(full) on seeds 2000-2009 vs 2010-2019 (two independent seed blocks).
"""
import json
import numpy as np

OUT = '/Users/esben/work/defense-1112-warning-words/swarm-defense-1112-warning-words/skeptic/out'
runs = json.load(open(f'{OUT}/vera_runs.json'))
AG = sorted({r['m'] for r in runs})
ARMS = ['neutral', 'honeypot', 'nameit', 'full', 'verbalise', 'trap']


def rates(rs):
    return {a: np.mean([np.mean([r['k'] for r in rs if r['m'] == m and r['a'] == a]) for m in AG]) for a in ARMS}


def G_of(rt):
    return (rt['neutral'] - rt['nameit']) - (rt['neutral'] - rt['honeypot'])


obs = rates(runs)
G_obs = G_of(obs)
rng = np.random.default_rng(2026)

# 1. honeypot/nameit permutation within agent
null = []
for _ in range(5000):
    acc = []
    for m in AG:
        v = [r['k'] for r in runs if r['m'] == m and r['a'] in ('honeypot', 'nameit')]
        lab = [r['a'] for r in runs if r['m'] == m and r['a'] in ('honeypot', 'nameit')]
        lab = rng.permutation(lab)
        hp = np.mean([k for k, l in zip(v, lab) if l == 'honeypot']); nm = np.mean([k for k, l in zip(v, lab) if l == 'nameit'])
        acc.append(hp - nm)
    null.append(np.mean(acc))
null = np.array(null)
p_two = float(np.mean(np.abs(null) >= abs(G_obs) - 1e-12))

# 2. full shuffle of all arm labels within agent
null_full, null_G, null_max = [], [], []
for _ in range(5000):
    rs = []
    for m in AG:
        sub = [r for r in runs if r['m'] == m]
        lab = rng.permutation([r['a'] for r in sub])
        rs += [dict(r, a=l) for r, l in zip(sub, lab)]
    rt = rates(rs)
    null_full.append(rt['neutral'] - rt['full']); null_G.append(G_of(rt))
    null_max.append(max(abs(rt['neutral'] - rt[a]) for a in ARMS[1:]))
cut_full_obs = obs['neutral'] - obs['full']
res = dict(G_obs=G_obs, perm_hp_name=dict(p_two=p_two, null_q95=float(np.quantile(np.abs(null), 0.95))),
           shuffle_all=dict(p_cut_full=float(np.mean(np.array(null_full) >= cut_full_obs)),
                            null_max_abs_cut_q95=float(np.quantile(null_max, 0.95)),
                            null_G_q95=float(np.quantile(np.abs(null_G), 0.95))))

# 3. split-half neutral
def half(par):
    return np.mean([np.mean([r['k'] for r in runs if r['m'] == m and r['a'] == 'neutral' and r['s'] % 2 == par]) for m in AG])
diffs = []
for _ in range(5000):
    d = []
    for m in AG:
        e = np.array([r['k'] for r in runs if r['m'] == m and r['a'] == 'neutral' and r['s'] % 2 == 0])
        o = np.array([r['k'] for r in runs if r['m'] == m and r['a'] == 'neutral' and r['s'] % 2 == 1])
        d.append(rng.choice(e, len(e)).mean() - rng.choice(o, len(o)).mean())
    diffs.append(np.mean(d))
res['split_half_neutral'] = dict(even=half(0), odd=half(1), diff=half(0) - half(1),
                                 ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])

# 4. seed halves
for name, lo, hi in (('seeds_2000_2009', 2000, 2009), ('seeds_2010_2019', 2010, 2019)):
    rt = rates([r for r in runs if lo <= r['s'] <= hi])
    res[name] = dict(G=G_of(rt), cut_full=rt['neutral'] - rt['full'], cut_hp=rt['neutral'] - rt['honeypot'],
                     cut_name=rt['neutral'] - rt['nameit'], cut_verbalise=rt['neutral'] - rt['verbalise'],
                     cut_trap=rt['neutral'] - rt['trap'])

json.dump(res, open(f'{OUT}/vera_null.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))
