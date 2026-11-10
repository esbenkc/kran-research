"""Iris: arrival size d (theory/transport.py) for the 5 rows-1-2-only random directions (u seeds 1112-1116) at
x matched to x_real 11.6948, computed BEFORE those cells finish, to freeze a rank prediction: larger d -> smaller T.
alpha solved by bisection on x as fed (bf16 cast once, rows 1-2 only), as rl.py --match-x does.
Run (from projects/cue-size-law): python3 -W ignore theory/transport_dirs.py -> theory/transport_dirs.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import transport as tp  # noqa: E402  (runs transport.py once; cheap)

XR = 11.6948


def x_of(on_rows):
    return float(np.linalg.norm(tp.N(on_rows) - tp.N(tp.h0)))


def solve(ufile, rows=(1, 2)):
    lo, hi = 1e-5, 1.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if x_of(tp.planted(mid, ufile, rows)) < XR:
            lo = mid
        else:
            hi = mid
    return hi


def d_med(on_rows):
    out = []
    for i in tp.idx:
        h = tp.bf16_to_f32(np.ascontiguousarray(tp.Z[i]))
        ho = h.copy(); ho[:3] = on_rows
        out.append(np.linalg.norm(tp.attn_row(ho)[0] - tp.attn_row(h)[0]))
    return float(np.median(out))


res = {}
for s in range(1112, 1117):
    f = f"data/cue_u_{s}.npy"
    a = solve(f)
    on = tp.planted(a, f, (1, 2))
    res[str(s)] = dict(alpha=a, x=round(x_of(on), 4), d_med=round(d_med(on), 5))
    print(s, res[str(s)], flush=True)
order = sorted(res, key=lambda k: -res[k]["d_med"])
json.dump(dict(rows12_xreal=res, predicted_T_order_fast_to_slow=order), open(Path(__file__).with_name("transport_dirs.json"), "w"), indent=1)
print("predicted order (fast -> slow):", order)
