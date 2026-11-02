"""Pre-data power at the frozen spec (Felix #12341/#12350): arms blocked, C1 (N=0), C4 (N=3), floor; 30/model.
Status on pooled C4 (equal-weight mean of 5): Yes = LB95 > 0.1025 and LB(C4-floor) > 0; No = UB95 < 0.1025.
Law: pred_C = floor + (blk-floor)(C-1)/C; d_C = p_C - pred_C; HOLD = both CI(d) contain 0, |d|<=0.10, LB(p4-p1)>0;
FAIL = any CI(d) excludes 0. Joint within-model x arm run bootstrap, B=2000, percentile 95%."""
import numpy as np, sys
blk = np.array([1.0, 0.875, 1.0, 0.975, 0.75]); n = 30; floor = np.zeros(5)
rng = np.random.default_rng(3); S, B = 300, 2000
def boot(k):  # k: counts (5,) -> bootstrap means (B,)
    return rng.binomial(n, k / n, size=(B, 5)).mean(1) / n
print("scenario: true C1 share, C4 share of blk | P(Yes) P(No) | law HOLD FAIL")
for s1, s4 in [(0, 0.05), (0, 0.15), (0, 0.3), (0, 0.5), (0, 0.75), (0.1, 0.75), (0.05, 0.6), (0.3, 0.6), (0.5, 0.8)]:
    y = no = h = f = 0
    for _ in range(S):
        kb = rng.binomial(n, blk); k1 = rng.binomial(n, blk * s1); k4 = rng.binomial(n, blk * s4); kf = rng.binomial(n, floor)
        bb, b1, b4, bf = boot(kb), boot(k1), boot(k4), boot(kf)
        lo4, hi4 = np.percentile(b4, [2.5, 97.5]); lod = np.percentile(b4 - bf, 2.5)
        y += lo4 > 0.1025 and lod > 0; no += hi4 < 0.1025
        d1 = b1 - bf; d4 = b4 - (bf + (bb - bf) * 0.75)
        c1 = np.percentile(d1, [2.5, 97.5]); c4 = np.percentile(d4, [2.5, 97.5])
        p1 = k1.mean() / n; p4 = k4.mean() / n; pb = kb.mean() / n; pf = kf.mean() / n
        fail = (c1[0] > 0 or c1[1] < 0) or (c4[0] > 0 or c4[1] < 0)
        hold = (not fail) and abs(p1 - pf) <= 0.10 and abs(p4 - pf - 0.75 * (pb - pf)) <= 0.10 and np.percentile(b4 - b1, 2.5) > 0
        h += hold; f += fail
    print(f"C1 {s1:.2f} C4 {s4:.2f} | {y/S:.2f} {no/S:.2f} | {h/S:.2f} {f/S:.2f}")
