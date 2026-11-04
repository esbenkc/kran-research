"""Per-seed D_add (one name seed only, cells of 2 runs) with its own within-scenario x arm shuffle null."""
import sys, numpy as np
sys.argv = ["x", "counted/vera_tags.jsonl", "key"]
import vera_rebuild as vr
for s in (0, 1, 2):
    sub = [r for r in vr.rows if r["seed"] == s]
    Y, ag, sc, arms, seeds = vr.table(sub); d = vr.fast_dadd(Y); rng = np.random.default_rng(20261104); null = []
    for i in range(2000):
        Z = Y.copy()
        for j in range(len(sc)):
            for a in range(len(arms)):
                Z[:, j, a] = rng.permutation(Y[:, j, a])
        null.append(vr.fast_dadd(Z))
    null = np.array(null)
    print(f"seed {s}: D_add {d:+.4f} null mean {null.mean():+.4f} q95 {np.quantile(null,.95):+.4f} p {(1+(null>=d).sum())/2001:.4f} use {Y.sum()}/{Y.size}")
for pair in ((0, 1), (0, 2), (1, 2)):
    sub = [r for r in vr.rows if r["seed"] in pair]
    print(f"seeds {pair}: D_add {vr.d_stats(sub)[0]:+.4f}")
