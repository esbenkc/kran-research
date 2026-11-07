# Vera: why build's C2 shuffle share (0.259) differs from mine (0.56): tie handling on a discrete null grid (step 1/115).
import numpy as np, sys
sys.argv = ['x', '../../kran-research/projects/scoped-recording/raw']
import vera_check as v
rows = v.load(sys.argv[1], ['none', 'key', 'flag'])
fo = v.cells(rows, 'flag_only')
obs = v.contrast_point(fo, 'key', 'flag')
for seed in (99, 1, 2):
    rng = np.random.default_rng(seed); nulls = []
    for s in range(2000):
        ds = []
        for m in sorted(fo):
            x = np.concatenate([fo[m]['key'], fo[m]['flag']]); rng.shuffle(x); ds.append(x[:23].mean() - x[23:].mean())
        nulls.append(np.mean(ds))
    nulls = np.array(nulls)
    print(f'seed {seed}: obs {obs:+.6f} (=3/115 {3/115:.6f}); share |null|>=|obs| exact-float {np.mean(np.abs(nulls) >= abs(obs)):.4f}, with 1e-12 tolerance {np.mean(np.abs(nulls) >= abs(obs) - 1e-12):.4f}, share |null|>=4/115 {np.mean(np.abs(nulls) >= 4/115 - 1e-12):.4f}; one-sided null>=obs {np.mean(nulls >= obs - 1e-12):.4f}')
