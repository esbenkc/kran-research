"""Extra checks on claims in #3206/#3208: per-token drop on the max14 direction (prefix vs answer),
fresh-probe catch vs C (the one knob that moved fresh 12.6% <-> 36.1%), and cosines of probe directions."""
import json
import sys
from pathlib import Path

import numpy as np

from vera_core import DATA  # noqa: E402
sys.argv = [sys.argv[0], str(DATA / "runs/lam0__seed*__tokens.npz")]
import vera_rebuild as R  # noqa: E402
from vera_core import Lin, auroc, thr  # noqa: E402

off, y, probe = R.off, R.y, R.probe
mx = R.mons["max14"][2]
wf = R.wf
out = {}
# shared prefix length again (twin by pair key)
tid = np.load(R.DATA / "clean_tokens.npz")["tok_id"]
idx = {}
for i, p in enumerate(R.pair):
    idx.setdefault(p, []).append(i)
pref = np.zeros(len(y), int)
for ii in idx.values():
    a, b = tid[off[ii[0]]:off[ii[0] + 1]], tid[off[ii[1]]:off[ii[1] + 1]]
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    pref[ii] = k
tokstmt = np.repeat(np.arange(len(y)), np.diff(off))
tokpos = np.arange(off[-1]) - off[tokstmt]
is_pref = tokpos < pref[tokstmt]
ev = np.isin(tokstmt, np.where(R.split == "eval")[0])
sc0 = R.clean[14] @ mx.w + mx.b
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
out["cos_frozen_max14"] = cos(wf, mx.w)
out["cos_frozen_last14"] = cos(wf, R.mons["last14"][2].w)
for f in R.files:
    att, _ = R.load(f)
    sd = Path(f).stem
    d = (att[14] @ mx.w + mx.b) - sc0
    r = {"drop_prefix_mean": float(d[ev & is_pref].mean()),
         "drop_answer_false_mean": float(d[ev & ~is_pref & (y[tokstmt] == 1)].mean()),
         "drop_answer_true_mean": float(d[ev & ~is_pref & (y[tokstmt] == 0)].mean())}
    Xa = R.seg_reduce(att[14], off, "mean")
    for C in (0.01, 0.1, 1.0, 10.0):
        lin = Lin(Xa[probe], y[probe], C)
        s = Xa @ lin.w + lin.b
        t = thr(s[R.refit_neg])
        r[f"fresh_C{C}"] = dict(catch=float((s[R.eval_pos] > t).mean()), auroc=float(auroc(s[R.eval_neg], s[R.eval_pos])))
        if C == 0.01:
            out.setdefault("cos_frozen_fresh", []).append(cos(wf, lin.w))
            out.setdefault("cos_max14_fresh", []).append(cos(mx.w, lin.w))
    out[sd] = r
    print(sd, json.dumps(r))
print({k: v for k, v in out.items() if k.startswith("cos")})
(Path(__file__).parent / "vera_extra.json").write_text(json.dumps(out, indent=1))
