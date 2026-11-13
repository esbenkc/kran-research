"""critic2: replicate cells (single twins, split halves) at t_h and T; min rank of reference rows before 256; A movement."""
import sys, os, json, numpy as np
sys.argv = [sys.argv[0], "."]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild import *
RB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild.json")))["cells"]
def rep_cell(k, t, pick):
    sp = RB["SM"][f"m{k}"][str(t)]["slice_p95"]
    tw = refs("SM", f"m{k}", k, t); v = top(resid(RUN[k], t, pick(tw)))[0][:, 0]
    et = e1(v, TRUE[k][None])[0]; en = e1(v, NUL)
    hm = max(e1(top(resid(f"nc_s{i}", t, pick(refs("SM", f"h{i}", k, t))))[0][:, 0], TRUE[k][None])[0] for i in range(6, 10))
    return bool(et > en.max() and et > hm and et > sp), 1 + int((en > et).sum())
out = {}
for name, pick in [("twin0", lambda x: x[:1]), ("twin1", lambda x: x[1:2]), ("twin2", lambda x: x[2:3]), ("twin3", lambda x: x[3:4]),
                   ("half1", lambda x: x[:2]), ("half2", lambda x: x[2:])]:
    for which, TT in [("t_h", TH), ("T", T)]:
        r = [rep_cell(k, TT[k], pick) for k in range(6)]
        out[f"{name}@{which}"] = dict(beats=sum(b for b, _ in r), ranks=[x for _, x in r])
minref = {a: min(RB[a][f"m{k}"][str(TT[k])]["rank"] for k in range(6) for TT in (TH, T)) for a in ["XS", "SS", "SM", "SMv", "SM0"]}
# A movement: SM twins start from the target's A_t; sm0 twins from the target's A0 -> sm0 twin at step t trained A; not A0.
print(json.dumps(dict(replicates=out, min_rank_reference_rows_t_h_T=minref), indent=1))
