"""critic2 (Claude fallback) independent rebuild of the tell-chain-8b headline cells.
Reads only data/chain/*.json (per-prompt dev lists), data/base_px.json and ref/parent_analysis.json.
Does not import analyze.py. Rule: PREREG.md section 3 (commit 92fa7c588)."""
import json, glob, math, os, sys
import numpy as np
from scipy import stats
P = sys.argv[1] if len(sys.argv) > 1 else '.'
base = json.load(open(f'{P}/data/base_px.json'))
SEEDS = [5, 6, 7, 8, 9]
def gapbase(s, cue):
    off = np.array(base['p_base_off'] if not isinstance(base['p_base_off'], dict) else base['p_base_off'][str(s)])
    on = np.array(base[f'p_base_{s}{cue}'])
    return off.mean() - on.mean()
def T_of(rec, cue, cap):
    gb = gapbase(rec['seed'], cue)
    cur = sorted(rec['curve'], key=lambda c: c['t'])
    Ds = [(c['t'], np.mean(c['p_off']) - np.mean(c[f'p_{cue}']) - gb) for c in cur if f'p_{cue}' in c]
    for i in range(len(Ds) - 1):
        if Ds[i][1] >= 0.10 and Ds[i + 1][1] >= 0.10 and Ds[i + 1][0] == Ds[i][0] + 8:
            return Ds[i][0], False, gb, Ds
    return cap, True, gb, Ds
R = {}
out = {'T': {}, 'cens': {}, 'gb_vs_json': 0.0}
for f in sorted(glob.glob(f'{P}/data/chain/*_s[0-9].json')):
    r = json.load(open(f))
    if r['kind'].startswith('trunk'):
        continue
    cue = r['cue']
    cap = 584 if r['kind'] == 'main_C' else 1536
    T, c, gb, Ds = T_of(r, cue, cap)
    out['gb_vs_json'] = max(out['gb_vs_json'], abs(gb - r['gap_base'][cue]))
    out['T'][r['name']] = T; out['cens'][r['name']] = c
    out.setdefault('T_json_mismatch', [])
    if T != r['T']:
        out['T_json_mismatch'].append((r['name'], T, r['T']))
def ci(num, den):
    lr = np.array([math.log(max(out['T'][f'{num}_s{s}'], 8) / max(out['T'][f'{den}_s{s}'], 8)) for s in SEEDS])
    m, se = lr.mean(), lr.std(ddof=1) / math.sqrt(len(lr))
    t = stats.t.ppf(0.975, len(lr) - 1)
    return dict(per_seed=[round(math.exp(x), 4) for x in lr], geo=math.exp(m), lo=math.exp(m - t * se), hi=math.exp(m + t * se), hi_ln=m + t * se, lo_ln=m - t * se)
res = {k: ci(*v) for k, v in {
    'RD_ctrl': ('main_D', 'ctrl_D'), 'RD_fresh': ('main_D', 'fresh_D'),
    'RC_ctrl': ('main_C', 'ctrl_C'), 'RC_fresh': ('main_C', 'fresh_C'),
    'dose_main_vs_dose1': ('main_C', 'dose1_C'), 'dose1_vs_ctrlC': ('dose1_C', 'ctrl_C'), 'dose1_vs_freshC': ('dose1_C', 'fresh_C'),
    'drift_C': ('ctrl_C', 'fresh_C'), 'drift_D': ('ctrl_D', 'fresh_D'), 'D_over_C_main': ('main_D', 'main_C')}.items()}
# parent stage B vs coin from parent raw T list
par = json.load(open(f'{P}/ref/parent_analysis.json'))
pT = {(x['arm'], x['seed']): x['T'] for x in par['runs'] if x.get('kB') == 4}
RB = {s: max(pT[('main', s)], 8) / max(pT[('coin', s)], 8) for s in SEEDS}
ln55 = math.log(0.55)
conf = all(res[k]['hi_ln'] < ln55 for k in ('RD_ctrl', 'RD_fresh'))
rej = any(res[k]['lo_ln'] > ln55 for k in ('RD_ctrl', 'RD_fresh'))
ncens = {k: sum(out['cens'][f'{k}_s{s}'] for s in SEEDS) for k in ('main_D', 'ctrl_D', 'fresh_D', 'main_C', 'ctrl_C', 'fresh_C', 'dose1_C')}
fall = [s for s in SEEDS if RB[s] > res['RC_ctrl']['per_seed'][SEEDS.index(s)] > res['RD_ctrl']['per_seed'][SEEDS.index(s)]]
fresh_med = float(np.median([out['T'][f'fresh_D_s{s}'] for s in SEEDS]))
freshC_med = float(np.median([out['T'][f'fresh_C_s{s}'] for s in SEEDS]))
# null cue E, zero-shot D(0)
dE = []; d0 = {}
for f in sorted(glob.glob(f'{P}/data/chain/*_s[0-9].json')):
    r = json.load(open(f))
    for c in r['curve']:
        if 'p_E' in c:
            dE.append((r['kind'], np.mean(c['p_off']) - np.mean(c['p_E']) - gapbase(r['seed'], 'E')))
    if not r['kind'].startswith('trunk'):
        c0 = min(r['curve'], key=lambda c: c['t'])
        d0.setdefault(r['kind'], []).append(np.mean(c0['p_off']) - np.mean(c0[f"p_{r['cue']}"]) - gapbase(r['seed'], r['cue']))
mainE = [v for k, v in dE if k.startswith('main')]; trE = [v for k, v in dE if k.startswith('trunk')]
summary = dict(T=out['T'], T_json_mismatch=out['T_json_mismatch'], gap_base_maxdiff=out['gb_vs_json'], censored=ncens,
               ratios={k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in res.items()},
               RB_coin_per_seed=RB, RB_coin_geo=math.exp(np.mean(np.log(list(RB.values())))),
               status='CONFIRMED' if conf and not rej else ('REJECTED' if rej else 'UNCONFIRMED'),
               port_gate_freshD_median=fresh_med, freshC_median=freshC_med,
               clause1=res['RD_ctrl']['geo'] <= 0.55, clause2_seeds_falling=fall,
               null_E=dict(main_min=min(mainE), main_max=max(mainE), n_main=len(mainE), trunk_min=min(trE), trunk_max=max(trE), n_trunk=len(trE)),
               D0={k: (min(v), max(v)) for k, v in d0.items()})
json.dump(summary, open(os.path.join(os.path.dirname(__file__), 'rebuild_critic2.json'), 'w'), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
