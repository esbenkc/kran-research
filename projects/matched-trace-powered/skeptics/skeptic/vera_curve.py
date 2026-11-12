"""Vera: exposure (summed text-difference mass) for both cues and the dev series, from the raw curve files only.
cue mass  = 0.10 x 32 probed rows x updates (logged cum_P_tv_mass, checked against the formula)
trace mass = trapezoid over dev checks of 32 x dev_D (rl.py logs no train-time trace TV)
"""
import json, sys
P = '/Users/esben/work/oracle-1112-matched-trace/kran-research/projects/matched-trace-powered/data/runs/'
rows = []
for f, off in [('joint_s0/curve.jsonl', 0), ('joint_s0_cont/curve.jsonl', 674)]:
    for l in open(P + f):
        d = json.loads(l)
        if 'dev_D' in d:
            d['g'] = d['step'] + off
            d['src'] = f.split('/')[0]
            rows.append(d)
rows.sort(key=lambda d: (d['g'], d['src']))
final = int(sys.argv[1]) if len(sys.argv) > 1 else None
# trace mass: trapezoid on (g, dev_D); at 674 both the parent's 650 row and the cont step-0 row exist
grid = []
for d in rows:
    if grid and grid[-1][0] == d['g']:
        grid[-1] = (d['g'], d['dev_D'])      # cont step-0 (the 674 adapter itself) replaces nothing; keep latest
    else:
        grid.append((d['g'], d['dev_D']))
mass = 0.0
for (g0, d0), (g1, d1) in zip(grid, grid[1:]):
    mass += 32 * (g1 - g0) * (d0 + d1) / 2
last_g, last_D = grid[-1]
out = {'grid_last': last_g, 'trace_mass_at_last_check': mass}
if final:
    out['final_update'] = final
    out['trace_mass_final_flat'] = mass + 32 * (final - last_g) * last_D
    out['cue_mass_final'] = 0.10 * 32 * final
    out['ratio_trace_over_cue'] = out['trace_mass_final_flat'] / out['cue_mass_final']
# logged cue mass vs formula on the cont rows
chk = [(d['g'], d.get('cum_P_tv_mass')) for d in rows if d['src'] == 'joint_s0_cont' and 'cum_P_tv_mass' in d]
out['cue_mass_check'] = [(g, m + 2156.8, 3.2 * g) for g, m in chk]
out['dev'] = [(d['g'], round(d['dev_forced_switch'], 4), round(d['dev_forced_switch'] / d['cheat_none'], 4),
               round(d['gap'], 4), round(d['dev_D'], 4), round(d['cheat_none'], 4)) for d in rows]
# planted dev share mean over continuation checks (local > 0)
cont = [d for d in rows if d['src'] == 'joint_s0_cont' and d['step'] > 0]
out['cont_mean_forced'] = sum(d['dev_forced_switch'] for d in cont) / len(cont)
out['cont_mean_sP_dev'] = sum(d['dev_forced_switch'] / d['cheat_none'] for d in cont) / len(cont)
par = [d for d in rows if d['src'] == 'joint_s0' and d['step'] >= 550]
out['parent_550_650_mean_forced'] = sum(d['dev_forced_switch'] for d in par) / len(par)
json.dump(out, open('vera_curve.json', 'w'), indent=1)
for k, v in out.items():
    if k != 'dev':
        print(k, v)
print('g, forced, sP_dev, trace_gap, dev_D, cheat_none')
for r in out['dev']:
    print(r)
