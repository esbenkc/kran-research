"""Hugo (skeptic2) null: can the re-set-line rule alone fake a 'hide'?
Score probes with NO falsehood signal on the clean states and on every run's attacked states (acts_150,
readout 29), with the same line rule (2nd-highest of the 200 refit-split true at that state):
  (a) 200 random unit directions in z (standardised with the clean refit-split mean/sd);
  (b) 20 logistic-regression refits on the 'probe' split with SHUFFLED labels (C=1, standardised z).
A null probe catches ~1% of false statements by construction (no signal). If attacked catch stays ~1%, the
rule does not create hides; the arm probes' fall from 94-98% to ~1% is then a real hide.
Also reports the truth-signal (unshuffled refit on the same split) as a positive control."""
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from check_runs import DATA, EVF, REF_T, TR

split = json.load(open(os.path.join(DATA, 'ref/split.json')))
SP = np.array([x['split'] for x in split])
z0 = np.load(os.path.join(DATA, 'clean/z_post.npy')).astype(np.float64)
mu, sd = z0[SP == 'refit'].mean(0), z0[SP == 'refit'].std(0) + 1e-6
rng = np.random.default_rng(7)
dirs = rng.standard_normal((1536, 200))
dirs /= np.linalg.norm(dirs, axis=0)
fit = SP == 'probe'
X = (z0[fit] - mu) / sd
probes = []
for k in range(20):
    y = rng.permutation(TR[fit])
    probes.append(LogisticRegression(C=1.0, max_iter=2000).fit(X, y).coef_[0])
pos = LogisticRegression(C=1.0, max_iter=2000).fit(X, 1 - TR[fit]).coef_[0]  # falsehood = positive class
W = np.concatenate([dirs, np.array(probes).T, pos[:, None]], 1)  # 200 random, 20 shuffled, 1 positive


def catches(z):
    S = ((z - mu) / sd) @ W
    line = np.sort(S[REF_T], 0)[::-1][1]
    return (S[EVF] > line).mean(0)


def summ(c):
    r, s, p = c[:200], c[200:220], c[220]
    return dict(random_median=float(np.median(r)), random_p95=float(np.percentile(r, 95)),
                shuffled_median=float(np.median(s)), shuffled_max=float(s.max()), positive=float(p))


out = {'clean': summ(catches(z0))}
for rdir in sorted(glob.glob(os.path.join(DATA, 'runs', '*__lam10000__seed*'))):
    f = os.path.join(rdir, 'acts_150.npy')
    if os.path.exists(f):
        A = np.load(f, mmap_mode='r')
        out[os.path.basename(rdir)] = summ(catches(np.asarray(A[:, 1, :], dtype=np.float64)))
for k, v in out.items():
    print(k.ljust(26), ' '.join(f'{a} {b:.3f}' for a, b in v.items()))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'null_probes.json'), 'w'), indent=1)
