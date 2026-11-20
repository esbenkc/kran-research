"""Vera: what-if for the missing main seed 2 (pre-data for seed 2). Pooled max18 catch + city-bootstrap CI over
seeds s0@150, s1@150 (build scores.npz) and a hypothetical seed 2 = (a) a copy of s0, (b) the CLEAN state (~99%,
the worst case for REJECTED). Shows whether the frozen status can still move once seed 2 is non-VOID."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vera_core as vc
D = Path(sys.argv[1])
M = vc.meta(D / "ref/split.json")
sc = lambda s: np.load(D / "states" / s / "scores.npz")["max"].astype(np.float64)
s0, s1, cl = sc("main__seed0__step150"), sc("main__seed1__step150"), sc("clean")
out = {}
for name, s2 in (("seed2=copy_s0", s0), ("seed2=clean_worst", cl)):
    L = [s0, s1, s2]
    pts = [vc.catch(x, M["y"], M["split"])["catch"] for x in L]
    b = vc.city_boot(L, M["y"], M["split"], M["city"], n=2000, seed=0)
    out[name] = dict(per_seed=pts, pooled=float(np.mean(pts)), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    print(name, out[name])
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
