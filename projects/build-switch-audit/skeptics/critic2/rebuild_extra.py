#!/usr/bin/env python3
"""critic2 (Claude restart) extra rebuild: norm ranks vs 50%-longer nulls, edits, nudge at unit 1-8, from raw JSON."""
import json
from pathlib import Path
P = Path(__file__).resolve().parents[2]; D = P / 'data'
SW = ['stack_s0','stack_s2','r50_s0','r50_s1']; CL = ['shared_penall_s0','sc_penall_s0','cham_r100_s0','cham_r100_s1']
n = json.load(open(D/'screen/norms.json'))
print('norm_switch=' + ','.join(f'{c}:{n[c]:.4f}' for c in SW))
print('norm_clean=' + ','.join(f'{c}:{n[c]:.4f}' for c in CL))
rr = json.load(open(D/'reading_rows.json'))
lon = {k: v['s1_fro_all'] for k, v in rr.items() if '@12288' in k}
print('norm_longer_12288=' + ','.join(f'{k}:{v:.4f}' for k, v in lon.items()))
for s in ['r50_s0','r50_s1','stack_s2','stack_s0']:
    above = [k for k, v in lon.items() if v > n[s]]
    print(f'longer_nulls_above_{s}({n[s]:.3f}): {len(above)}/3 {above}')
q = {k: v['s2_sigma1_q27'] for k, v in rr.items() if '@12288' in k}
print('s2_longer=' + ','.join(f'{k}:{v:.3f}' for k, v in q.items()))
ed = json.load(open(D/'edits.json'))
for s, m in ed['sessions'].items():
    rows = [(r['cid'], 100*r['gap_before'], 100*r['gap_after'], 100*r['fraction'], r.get('E1')) for r in m.values() if r.get('edit')]
    passed = sorted(c for c, *_, e in rows if e)
    fr = [x[3] for x in rows]
    print(f'edit {s}: E1 pass={passed} frac={min(fr):.4f}-{max(fr):.4f}% ' + ' '.join(f'{c}:{a:.1f}->{b:.1f}' for c, a, b, _, _ in rows))
for c, v in ed['controls'].items():
    print(f'random_head_control {c}: median gap {100*v["median_gap"]:.1f}')
V = P/'skeptics/skeptic'
b = json.load(open(V/'vera_app_base_v16.json'))['base']; pn = json.load(open(V/'vera_app_panel_v16.json'))
for c in SW + CL:
    r = pn[c]
    print(f'nudge_unit1-8 {c}: {100*r["p_eager"]:.1f}% -> {100*r["p_pert"][1]:.1f}% (delta {100*(r["p_pert"][1]-r["p_eager"]):+.2f} pts); max delta any unit 1-4 {max(100*(x-r["p_eager"]) for x in r["p_pert"][1:]):+.2f}')
