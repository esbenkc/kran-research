"""critic2: turn rebuild_critic2.json + own null-E pass into rebuild_cells.json (cell names = data/cells stems)."""
import json, glob, math, os, sys
import numpy as np
P = sys.argv[1]; H = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(f'{H}/rebuild_critic2.json'))
base = json.load(open(f'{P}/data/base_px.json'))
par = json.load(open(f'{P}/ref/parent_analysis.json'))
pT = {(x['arm'], x['seed']): x['T'] for x in par['runs'] if x.get('kB') == 4}
S = [5, 6, 7, 8, 9]; cells = []
add = lambda c, v: cells.append({'cell': c, 'value': v})
for name, T in r['T'].items():
    add(f'T_{name}', T)
for key, cell in [('RD_ctrl', 'RD_ctrl'), ('RD_fresh', 'RD_fresh'), ('RC_ctrl', 'RC_ctrl'), ('RC_fresh', 'RC_fresh')]:
    for i, s in enumerate(S):
        add(f'{cell}_s{s}', r['ratios'][key]['per_seed'][i])
    add(f'{cell}_seedlevel', r['ratios'][key]['geo'])
for s in S:
    add(f'RB_ctrl_s{s}', round(pT[('main', s)] / pT[('coin', s)], 4))
    add(f'RB_fresh_s{s}', round(pT[('main', s)] / pT[('fresh', s)], 4))
add('dose_main_vs_dose1', r['ratios']['dose_main_vs_dose1']['geo'])
add('dose_dose1_vs_ctrl', r['ratios']['dose1_vs_ctrlC']['geo'])
for arm, kinds in [('main', ('main_C', 'main_D')), ('trunk', ('trunk1', 'trunk2'))]:
    for s in S:
        gb = np.mean(base['p_base_off']) - np.mean(base[f'p_base_{s}E'])
        v = 0.0
        for k in kinds:
            rec = json.load(open(f'{P}/data/chain/{k}_s{s}.json'))
            for c in rec['curve']:
                if 'p_E' in c:
                    v = max(v, abs(np.mean(c['p_off']) - np.mean(c['p_E']) - gb))
        add(f'null_cue_E_{arm}_s{s}', round(v, 5))
json.dump({'by': 'critic2 (Claude fallback, claude-opus-5-5)', 'cells': cells}, open(f'{H}/rebuild_cells.json', 'w'), indent=1)
# compare with the team's cells
mism = []
for c in cells:
    f = f"{P}/data/cells/{c['cell']}.json"
    if not os.path.exists(f):
        mism.append((c['cell'], 'NO CELL FILE')); continue
    tv = json.load(open(f))['value']
    tol = 0 if isinstance(c['value'], int) else 6e-4
    if abs(float(tv) - float(c['value'])) > tol:
        mism.append((c['cell'], c['value'], tv))
print(len(cells), 'cells;', 'mismatch:', mism)
