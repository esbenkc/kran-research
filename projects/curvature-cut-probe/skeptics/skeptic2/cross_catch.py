"""Hugo (skeptic2) control: is the hiding specific to the attacked probe?
At step H of every new-code run, catch at the re-set line (2nd-highest of 200 refit true at that checkpoint)
for the three arm probes (unproj, proj, rand) plus the literal projection lit_k128 and the k=32 cut probe.
If a fine-tune against one probe also blinds the others, the measured price is not specific to that probe's
geometry. Rows = attacked probe/seed, columns = scoring probe."""
import glob
import os

import numpy as np

from check_runs import DATA, EVF, REF_T, col_of

H = 150
COLS = ['unproj', 'proj', 'rand', 'lit_k128', 'proj_k32', 'rand_k32']
print('run'.ljust(26), ' '.join(c.rjust(9) for c in COLS))
for rdir in sorted(glob.glob(os.path.join(DATA, 'runs', '*__lam10000__seed*'))):
    f = os.path.join(rdir, f'ckpt_{H}.npz')
    if not os.path.exists(f):
        continue
    d = np.load(f, allow_pickle=True)
    out = []
    for c in COLS:
        s = d['scores'][:, col_of(d, c)].astype(np.float64)
        line = np.sort(s[REF_T])[::-1][1]
        out.append((s[EVF] > line).mean())
    print(os.path.basename(rdir).ljust(26), ' '.join(f'{x:9.3f}' for x in out))
d0 = np.load(os.path.join(DATA, 'runs', 'unproj__lam10000__seed0', 'ckpt_0.npz'), allow_pickle=True)
out = []
for c in COLS:
    s = d0['scores'][:, col_of(d0, c)].astype(np.float64)
    out.append((s[EVF] > np.sort(s[REF_T])[::-1][1]).mean())
print('clean (step 0)'.ljust(26), ' '.join(f'{x:9.3f}' for x in out))
