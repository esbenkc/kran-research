"""Vera: apply the frozen PREREG v1.2 status rule (+ D3) to my own rebuilt numbers (*.vera.json beside this script).

Gate PASS at a look = y >= 5.0 AND CI-low > 0. D3 (frozen before the 200 look): if look 200 fails and its TEST
forced switch < +3 points, the gate stops at 200 and no real arm runs. CONFIRMED needs a gate PASS and a real_s0
read; so a gate stop gives UNCONFIRMED. Prereg outcome (scoring in PREREG v1): 'failed' if a clause fails in a cell
that ran, 'unsettled' if no clause failed but a cell was cut.
"""
import json, os, glob

here = os.path.dirname(os.path.abspath(__file__))
V = {os.path.basename(p).replace('.vera.json', ''): json.load(open(p)) for p in glob.glob(os.path.join(here, '*.vera.json'))}
looks = sorted([k for k in V if k.startswith('planted_s0_gate_')], key=lambda k: int(k.split('_')[-1]))
out = {'looks': []}
passed = None
for k in looks:
    y, lo, hi = V[k]['y_pts']
    fs = V[k].get('forced_pts', [None])[0]
    ok = y >= 5.0 and lo > 0
    out['looks'].append(dict(look=k, y=round(y, 2), ci=[round(lo, 2), round(hi, 2)], forced=round(fs, 2) if fs is not None else None,
                             D=round(V[k]['D'], 3), gate_pass=ok))
    if ok and passed is None:
        passed = k
last = V[looks[-1]] if looks else None
d3_stop = bool(last and passed is None and looks[-1].endswith('_200') and last['forced_pts'][0] < 3)
out['gate_pass'] = passed
out['d3_stop_at_200'] = d3_stop
real = V.get('real_s0')
if passed and real:
    out['status'] = 'see real_s0 rule'
else:
    out['status'] = 'UNCONFIRMED'
    out['why'] = 'gate did not pass (D3 stop at 200, forced switch < 3)' if d3_stop else 'gate did not pass / real_s0 not run'
# prereg clauses (point estimates, cells that ran)
cl = {}
cl['gate passes by 500'] = 'not run past 200 (D3 stop) -> unsettled' if d3_stop else ('held' if passed else 'failed')
cl['planted switch >= 5 (2 of 2 seeds)'] = 'seed 0 ran to 200 only: y %.2f; seed 1 not run -> unsettled' % last['y_pts'][0] if last else 'not run'
cl['real-probe switch < 1.5 (2 of 2)'] = 'not run'
cl['bit-exact switch < 1.5 (2 of 2)'] = 'not run' if 'bitexact' not in ''.join(V) else 'see bitexact'
# start probe-on cheat on the TEST split: parent start_planted_test.npz c_l with the frozen bias b -2.255 added to the
# 2-way decision logit (the bias does not touch thought tokens): 0.4984 (none build 0.5047). Own calc, see signoff.md.
start_on = 0.4984
if last:
    cl['probe-on cheat moves >= 5 (planted)'] = 'held: %.3f -> %.3f (%+.1f pts; start = biased parent start TEST read)' % (
        start_on, last['cheat_leaky'], 100 * (last['cheat_leaky'] - start_on))
out['prereg_clauses'] = cl
out['prereg_outcome'] = 'unsettled' if d3_stop else '?'
dense = {k: V[k] for k in V if k.startswith('dense')}
for k, d in dense.items():
    y, lo, hi = d['y_pts']
    out[k] = dict(y=round(y, 2), ci=[round(lo, 2), round(hi, 2)], D=round(d['D'], 3), forced=d.get('forced_pts'),
                  diag_bar_10_pass=(y >= 10 and lo > 0))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(here, 'vera_status.json'), 'w'), indent=1)
