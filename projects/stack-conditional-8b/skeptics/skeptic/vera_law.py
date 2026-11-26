"""Vera: cue-size law (cue-size-law-8b-rep, seeds 5-7) -> predicted onset T for a given share.
share = ||N(h+d) - N(h)|| / ||N(h)|| over rows 0-2 (block-34 input RMSNorm), as a fraction.
Usage: python3 vera_law.py [share ...]"""
import json, math, sys, os
d = json.load(open(os.path.join(os.path.dirname(__file__), "rep_analysis.json")))
ps = d["status_8b"]["per_seed"]
fits = [(v["slope"], v["intercept"]) for v in ps.values() if v.get("slope") is not None]
def T(share, b, a):
    return math.exp(a + b * math.log(share))
def share_for(Tt, b, a):
    return math.exp((math.log(Tt) - a) / b)
print("per-seed fits (slope, intercept):", [(round(b, 4), round(a, 4)) for b, a in fits])
for b, a in fits:
    print(f"  slope {b:.3f}: T(5.469%)={T(0.05469,b,a):.0f}  share at T=8192: {100*share_for(8192,b,a):.4f}%  at T=4096: {100*share_for(4096,b,a):.4f}%")
for s in [float(x) for x in sys.argv[1:]]:
    ts = [T(s, b, a) for b, a in fits]
    print(f"share {100*s:.5f}% -> T per seed {[round(t) for t in ts]}  (2x band {min(ts)/2:.0f}..{2*max(ts):.0f})")
