"""Knock-out null (own, post-data reading, not status): copy cue-off keys in ALL pairs EXCEPT a 3-set S.
Necessity of S = share(G) - share(all but S). Compares BAND's necessity with 3-sets from 28-60 (my VN draws, first 100)
and uniform 3-sets (my VU). Usage: python vera_ko.py SEED
"""
import sys, json
import numpy as np
sys.path.insert(0, "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic")
import vera_pe as V

seed = sys.argv[1]
mine = json.load(open(f"{V.HERE}/vera_rows_{seed}_fp32.json"))
E = V.Eng(seed, "fp32", "cpu"); H = E.CON; don = E.donor()
on, off = E.px(H), E.px(E.COFF); gap = off.mean() - on.mean()
def ko(S):
    keep = [p for p in range(64) if p not in S]
    p = E.px(H, mode="patch", blocks=[0, 1], donor=don, dims=V.pair_dims(keep), pos=[0, 1, 2])
    return float((p.mean() - on.mean()) / gap)
G = ko([])
res = {"G": G, "BAND_nec": G - ko([61, 62, 63])}
vn = [G - ko(s) for s in mine["sets"]["VN"][:100]]
vu = [G - ko(s) for s in mine["sets"]["VU"]]
res.update(VN_nec=vn, VU_nec=vu,
           VN_nec_median_p95_max=[float(np.median(vn)), float(np.percentile(vn, 95)), float(np.max(vn))],
           VU_nec_median_p95_max=[float(np.median(vu)), float(np.percentile(vu, 95)), float(np.max(vu))],
           BAND_rank_among_VN=int(1 + sum(v >= res["BAND_nec"] for v in vn)))
json.dump(res, open(f"{V.HERE}/vera_ko_{seed}.json", "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("VN_nec", "VU_nec")})
