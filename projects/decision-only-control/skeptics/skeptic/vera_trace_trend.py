"""Vera: is the trace starting to be learned late in d010_s0? Paired (same prompt resamples) contrast of the
mean trace share over late reads vs early reads, plus end-minus-start. Own code (vera_check.py estimators)."""
import json, numpy as np, vera_check as V
T = V.load_test(); bt = V.Boot(T['prompt'])
run = 'd010_s0'; steps = V.reads_of(run)
S = {s: V.stats(T, np.load(f'{V.P}/data/runs/{run}/reads/c_{s}.npy').astype(np.float64), bt) for s in steps}
early = [s for s in steps if s <= 500]; late = [s for s in steps if s >= 1100]
e = np.mean([S[s]['s_R'] for s in early], 0); l = np.mean([S[s]['s_R'] for s in late], 0)
out = dict(early_reads=early, late_reads=late, early=V.pc(e), late=V.pc(l), late_minus_early=V.pc(l - e),
           end_minus_start=V.pc(S[steps[-1]]['s_R'] - S[0]['s_R']),
           y_R_end_pts=V.pc(100 * S[steps[-1]]['y_R']), y_R_start_pts=V.pc(100 * S[0]['y_R']),
           s_R_by_read={s: round(float(S[s]['s_R'][0]), 4) for s in steps})
json.dump(out, open('vera_trace_trend.json', 'w'), indent=1); print(json.dumps(out))
