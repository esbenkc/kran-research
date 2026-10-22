"""Iris: Tove's line lemma for a max-panel with ONE alarm line (#4933).
Lemma: for M = max_l z_l and t(.) = k-th highest over the same true statements, t(M) >= max_l t(z_l), because
M >= z_l pointwise and the k-th order statistic is monotone. Union bounds:
max_l P(z_l(false) > t_M) <= catch(M) = P(max_l z_l(false) > t_M) <= sum_l P(z_l(false) > t_M).
(1) property test on 20,000 random panels (Gaussian, heavy-tailed and shifted members, n = 200, k = 2);
(2) per state: t_M, the member whose own line equals t_M (if any), each member's catch at its own line and at t_M.
Run from projects/panel-fine-tune: ../../../venv1021/bin/python iris_checks/line_lemma.py OUT.json"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from common import pool_idx, rows, threshold

rng = np.random.default_rng(1022)
viol = 0
for trial in range(20000):
    L = rng.integers(2, 25)
    T = rng.standard_t(df=rng.integers(1, 30), size=(200, L)) * rng.uniform(.2, 3, L) + rng.normal(0, 2, L)
    F = rng.standard_t(df=5, size=(300, L)) * rng.uniform(.2, 3, L) + rng.normal(0, 4, L)
    tM = threshold(T.max(1)); tl = np.array([threshold(T[:, j]) for j in range(L)])
    c = (F.max(1) > tM).mean(); per = (F > tM).mean(0)
    viol += int(tM < tl.max() - 1e-12) + int(c < per.max() - 1e-12) + int(c > per.sum() + 1e-12)
out = {'property_test_trials': 20000, 'violations': viol}
Z = json.load(open('data/fits/zmoments.json')); P = list(range(10, 30))
_, y, _, _ = rows(); tr, li, ev = pool_idx(); evf = ev[y[ev] == 1]
import glob, os
states = ['clean'] + sorted(os.path.basename(p)[:-4] for p in glob.glob('data/scores/panel__*.npz'))
for st in states:
    S = np.load(f'data/scores/{st}.npz')
    A = np.stack([(S[f'tok{R}'] - Z[f'tok{R}']['mu']) / Z[f'tok{R}']['sd'] for R in P], 1)
    M = A.max(1); tM = threshold(M[li]); tl = {R: threshold(A[li, j]) for j, R in enumerate(P)}
    own = {R: float((A[evf, j] > tl[R]).mean()) for j, R in enumerate(P)}
    atM = {R: float((A[evf, j] > tM).mean()) for j, R in enumerate(P)}
    c = float((M[evf] > tM).mean())
    out[st] = dict(t_M=tM, max_member_line=max(tl.values()), argmax_member_line=int(max(tl, key=tl.get)),
                   catch_M=c, lower=max(atM.values()), upper=sum(atM.values()),
                   own_line_catch={R: round(v, 4) for R, v in own.items()},
                   catch_at_t_M={R: round(v, 4) for R, v in atM.items()})
    print(st, 't_M %.3f max member line %.3f (tok%d) catch_M %.4f bounds [%.4f, %.4f] tok11 own %.3f at t_M %.4f' % (
        tM, max(tl.values()), max(tl, key=tl.get), c, max(atM.values()), sum(atM.values()), own[11], atM[11]))
print('property test violations:', viol)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
