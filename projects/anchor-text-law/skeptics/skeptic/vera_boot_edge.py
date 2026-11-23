# Vera: the B-raw status cell sits at the REJECTED boundary (hi vs -1/10). How stable is the 95% upper end
# across bootstrap seeds and percentile conventions? Fast numpy version of the frozen resampling
# (resample runs within B-raw, B-md, B-anchor, A-raw; err = catch(B,raw) - reach_raw(A) * follow_B).
import sys, numpy as np, vera_check as V
a, b = V.model_dirs()
evA, _, _ = V.build(a[0]); evB, _, _ = V.build(b[0])
nA, nB = V.scored_n(evA), V.scored_n(evB)
def arr(ev, c, n):
    return np.array([[e['reach'], e['follow'], e['reach'] and e['follow']] for e in (ev[c][i] for i in range(n))], dtype=float)
Braw, Bmd, Ban, Araw = arr(evB, 'raw', nB), arr(evB, 'md', nB), arr(evB, 'anchor', nB), arr(evA, 'raw', nA)
def draws(seed, D=2000):
    r = np.random.default_rng(seed)
    pick = lambda X: X[r.integers(0, len(X), (D, len(X)))]
    br, bm, ba, ar = pick(Braw), pick(Bmd), pick(Ban), pick(Araw)
    followB = (bm[:, :, 2].sum(1) + ba[:, :, 2].sum(1)) / (bm[:, :, 0].sum(1) + ba[:, :, 0].sum(1))
    return br[:, :, 1].mean(1) - ar[:, :, 0].mean(1) * followB
his = {'np.percentile linear': [], 'lower-index (sorted[int(.975D)-1])': [], 'higher (sorted[ceil(.975D)])': []}
for s in range(500):
    e = np.sort(draws(s))
    his['np.percentile linear'].append(np.percentile(e, 97.5))
    his['lower-index (sorted[int(.975D)-1])'].append(e[int(0.975 * 2000) - 1])
    his['higher (sorted[ceil(.975D)])'].append(e[1950])
for k, v in his.items():
    v = np.array(v)
    print(f'{k:36s} hi over 500 seeds: median {np.median(v):+.4f} min {v.min():+.4f} max {v.max():+.4f}  P(hi < -0.10) {np.mean(v < -0.1):.3f}')
big = np.sort(draws(1, 200000))
print('200k-draw percentile 97.5: %+.4f, 2.5: %+.4f' % (np.percentile(big, 97.5), np.percentile(big, 2.5)))
print('P(err_boot >= -0.10) in 200k draws: %.4f (REJECTED needs < 0.025)' % np.mean(big >= -0.1))
