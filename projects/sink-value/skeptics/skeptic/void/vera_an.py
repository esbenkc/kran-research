"""Shares + paired prompt-bootstrap CIs from my own vera_kv*.npz files. Usage: python vera_an.py FILE.npz [FILE2 ...]"""
import sys, numpy as np
def share(z, k, on, off):
    return (z[k].mean() - on.mean()) / (off.mean() - on.mean())
for f in sys.argv[1:]:
    z = np.load(f); on, off = z["on"], z["off"]
    rng = np.random.default_rng(7); B = rng.integers(0, len(on), (4000, len(on)))
    print(f"== {f}: P on {on.mean():.4f} off {off.mean():.4f}  noop max|dP| {np.abs(z['noop'] - on).max():.2e}")
    if "resid@0-2" in z and "kv@0-2" in z:
        print(f"   gate resid@0-2 {share(z, 'resid@0-2', on, off):.4f} vs kv@0-2 {share(z, 'kv@0-2', on, off):.4f}  max|dP| {np.abs(z['resid@0-2'] - z['kv@0-2']).max():.2e}")
    for k in z.files:
        if k.startswith(("on_", "off_")) or k in ("on", "off") or z[k].ndim != 1: continue
        if "h" in k.split("_")[-1] and k.split("_")[-1][0] == "L": continue
        d = z[k]; s = share(z, k, on, off)
        bs = (d[B].mean(1) - on[B].mean(1)) / (off[B].mean(1) - on[B].mean(1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {k:22s} {s:7.3f} [{lo:6.3f}, {hi:6.3f}]")
    for which in ("v", "k"):
        row = []
        for L in (0, 1):
            for h in range(16):
                kk = f"{which}@0-2_L{L}h{h}"
                if kk in z: row.append(share(z, kk, on, off))
        if row: print(f"   per-head {which}@0-2: " + " ".join(f"{x:.2f}" for x in row))
    for L in (0, 1):
        for nm in ("off", "on"):
            for q in ("ans", "all"):
                k = f"{nm}_{q}_L{L}"
                if k in z: a = z[k]; print(f"   att {k}: tok0/1/2 mean {a.mean(0).round(4)}  heads >0.5 on 0-2: {int((a.sum(1) > 0.5).sum())}/16  per-head sum: {' '.join(f'{x:.2f}' for x in a.sum(1))}")
