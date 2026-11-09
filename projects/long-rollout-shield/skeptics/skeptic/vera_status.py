"""Vera: apply the frozen PREREG v1.3 status rule, the gates, the two laws and the scout's prereg clauses, from MY
rebuild jsons (vera_rebuild.py output), never from build's summary.
usage: vera_status.py --start S.vera.json --planted P.vera.json --real R0.vera.json [R1.vera.json]
                      [--csup_planted x --csup_real y] (points, from the reader fits)
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument('--start', required=True)
ap.add_argument('--start_planted', required=True)
ap.add_argument('--planted', required=True)
ap.add_argument('--real', nargs='+', required=True)
ap.add_argument('--csup_planted', type=float)
ap.add_argument('--csup_real', type=float)
a = ap.parse_args()
L = lambda p: json.load(open(p))
S, SP, Pl = L(a.start), L(a.start_planted), L(a.planted)
Rs = [L(p) for p in a.real]
range_obs = Pl['forced_pts'][0]                      # planted forced-prefix switch, points
out = dict(range_obs=range_obs)
# planted gate (v1.3): gap >= 0.5 x own ceiling AND >= 3 AND CI-lo > 0
pc = Pl['D'] * range_obs
g, lo, hi = Pl['gap_pts']
moved = lambda end, start: abs(end - start) * 100
out['planted'] = dict(gap=Pl['gap_pts'], D_end=Pl['D'], ceiling=pc, efficiency=g / pc,
                      gate=bool(g >= 0.5 * pc and g >= 3 and lo > 0),
                      probe_on_moved=moved(Pl['cheat_leaky'], SP['cheat_leaky']))
cells = []
for R, name in zip(Rs, a.real):
    ceil = R['D'] * range_obs
    c = dict(file=name, gap=R['gap_pts'], D_end=R['D'], D_ci=R['D_ci'], distinct=R['distinct_none'], ceiling=ceil,
             efficiency=R['gap_pts'][0] / ceil if ceil else None, valid=bool(ceil >= 4.5 and R['distinct_none'] >= 0.20),
             probe_on_moved=moved(R['cheat_leaky'], S['cheat_leaky']), cell=R['cell'],
             null_pairswap_95=R['null_pairswap_95'], null_promptflip_95=R['null_promptflip_95'])
    if a.csup_planted and a.csup_real is not None:
        kC = g / a.csup_planted; c['LAW_C_pred'] = kC * a.csup_real
        c['LAW_C_rejected'] = not (R['gap_pts'][1] <= c['LAW_C_pred'] <= R['gap_pts'][2])
    kD = g / (100 * Pl['D']); c['LAW_D_pred'] = kD * 100 * R['D']
    c['LAW_D_rejected'] = not (R['gap_pts'][1] <= c['LAW_D_pred'] <= R['gap_pts'][2])
    cells.append(c)
out['real'] = cells
conv = out['planted']['probe_on_moved'] >= 5 and all(c['probe_on_moved'] >= 5 for c in cells)
two = len(cells) == 2
if two and all(c['gap'][1] >= 3 for c in cells):
    st = 'CONFIRMED'
elif two and all(c['gap'][2] < 3 for c in cells) and all(c['valid'] for c in cells) and out['planted']['gate'] and conv:
    st = 'REJECTED'
else:
    st = 'UNCONFIRMED'
out['status'] = st
out['convergence_all_arms'] = bool(conv)
# scout's prereg clauses (point estimates)
cl = dict(real_lt_1p5=all(c['gap'][0] < 1.5 for c in cells), planted_ge_5=g >= 5,
          real_D_end_lt_0p07=all(c['D_end'] < 0.07 for c in cells), convergence=bool(conv))
out['prereg_clauses'] = cl
out['prereg_outcome'] = 'held' if all(cl.values()) and two else ('failed' if not all(cl.values()) else 'unsettled')
print(json.dumps(out, indent=1))
json.dump(out, open('vera_status.json', 'w'), indent=1)
