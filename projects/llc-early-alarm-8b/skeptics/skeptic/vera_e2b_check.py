"""Vera: rebuild E2b (k2 s5 u88, c 0.1, gamma 1, 2 chains x 3,200 steps, burn 800) from raw per-step losses.
Reports: lambda-hat (draws 801-3200, both chains), batch-means SE (5 batches), the running estimate n beta * mean of
the minibatch loss in 100-step windows (no smoothing), and Felix's frozen test: is the estimate flat (within its
batch-means SE) from ~1,600 steps on? Flat = the last 1,600 steps' first-half and second-half means differ by
<= 2 SE (sec 2b rule, scaled to the long walk)."""
import json, math, sys, glob
import numpy as np
f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(
    "/Users/esben/work/theory-1124-llc-early-alarm-8b/kran-research/projects/llc-early-alarm-8b/data/llc/e2b/*.json"))[-1]
r = json.load(open(f))
print("file", f, "keys", [k for k in r if k != "losses"])
L = np.array(r["losses"], float)
nb = r["nbeta"]; burn = r.get("burn", 800); Ls = r.get("L_wstar_all_rows", 0.0)
print("losses shape", L.shape)
d = L[:, burn:]
lam = nb * (d.mean() - Ls)
pooled = d.reshape(-1)
bm = np.array([nb * (b.mean() - Ls) for b in np.array_split(pooled, 5)])
se = bm.std(ddof=1) / math.sqrt(5)
print(f"lambda-hat {lam:.3f} (stored {r.get('lambda_hat')}), batch-means SE {se:.3f}")
win = [nb * (L[:, a:a + 100].mean() - Ls) for a in range(0, L.shape[1], 100)]
print("window means per 100 steps (both chains):", [round(x, 1) for x in win])
for ch in range(L.shape[0]):
    print(f" chain {ch} per 400 steps:", [round(nb * (L[ch, a:a + 400].mean() - Ls), 1) for a in range(0, L.shape[1], 400)])
late = L[:, 1600:]
h1, h2 = nb * late[:, :late.shape[1] // 2].mean(), nb * late[:, late.shape[1] // 2:].mean()
seg = np.array([nb * b.mean() for b in np.array_split(late.reshape(-1), 5)]); se_l = seg.std(ddof=1) / math.sqrt(5)
print(f"steps 1601-2400 vs 2401-3200: {h1:.2f} vs {h2:.2f}, diff {h2 - h1:+.2f}, SE(late) {se_l:.2f} -> flat: {abs(h2 - h1) <= 2 * se_l}")
eps = r.get("eps")
if eps and "dist_from_wstar" in r:
    dist = np.array(r["dist_from_wstar"], float)
    print("dist samples (first chain, first/last 4):", dist[0][:4], dist[0][-4:])
