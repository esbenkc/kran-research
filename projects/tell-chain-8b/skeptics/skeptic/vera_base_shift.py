# Sensitivity: shift every untrained-tail base gap by delta (my forward pass gave the s5 D base gap 0.0047 lower than
# the logged one; per-prompt precision noise is ~0.005 in the mean). Recompute all T and the frozen status.
import sys, math
import numpy as np
import vera_chain as vc
d = sys.argv[1]
recs, meta = vc.load(d)
orig = dict(vc.FRESH_BASE)
for delta in (-0.01, -0.005, 0.0, 0.005, 0.01):
    # shift = move p_cue lists of the base by +delta  (gap_base -> gap_base - delta)
    vc.FRESH_BASE.clear()
    for k, (off, on) in orig.items():
        vc.FRESH_BASE[k] = (off, list(np.asarray(on) + delta))
    Ts = vc.all_T(recs, meta)
    st, res = vc.status(Ts, True)
    tD = [Ts[('main_D', s)]['T'] for s in vc.SEEDS]; tc = [Ts[('ctrl_D', s)]['T'] for s in vc.SEEDS]; tf = [Ts[('fresh_D', s)]['T'] for s in vc.SEEDS]
    print(f"gap_base shift {-delta:+.3f}: {st}  R_D ctrl {res['ctrl_D']['geo']:.3f} [{res['ctrl_D']['lo']:.3f}, {res['ctrl_D']['hi']:.3f}]  "
          f"fresh {res['fresh_D']['geo']:.3f} [{res['fresh_D']['lo']:.3f}, {res['fresh_D']['hi']:.3f}]  T main_D {tD} ctrl_D {tc} fresh_D {tf}")
