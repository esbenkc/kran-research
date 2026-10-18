"""Vera: rebuild the probe-fit numbers from data/probe_fit_scores.npz (raw calib scores only).

Own AUROC (Mann-Whitney with tie handling), own 5%-FPR thresholds, own ensemble max,
shuffled-label null for AUROC.
"""
import numpy as np, json, sys

P = "/Users/esben/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/"
z = np.load(P + "probe_fit_scores.npz")
met = json.load(open(P + "probe_fit_metrics.json"))
y = z["y_calib"].astype(int)


from vera_stats import auroc, thr_at_fpr


members = [k[len("calib_"):] for k in z.files if k.startswith("calib_L")]
print(f"n calib = {len(y)}  (deceptive {y.sum()}, truthful {len(y)-y.sum()})")
print(f"{'member':<12} {'AUROC':>7} {'build':>7} {'thr5':>6} {'build':>6} {'mean T':>7} {'mean D':>7} {'gap':>6}")
M = np.stack([z["calib_" + m] for m in members]).astype(float)
for m, s in zip(members, M):
    b = met["members"][m]
    print(f"{m:<12} {auroc(s,y):7.4f} {b['auroc_calib_group2']:7.4f} {thr_at_fpr(s,y):6.3f} {b['thr_5fpr']:6.3f} "
          f"{s[y==0].mean():7.3f} {s[y==1].mean():7.3f} {s[y==1].mean()-s[y==0].mean():6.3f}")

ens = M.max(0)
print("\nensemble max == saved calib_ensemble:", np.allclose(ens, z["calib_ensemble"], atol=1e-6, rtol=0))
thr = np.array([met["members"][m]["thr_5fpr"] for m in members])
flag = (M > thr[:, None]).any(0)
A = z["calib_L12_mean"].astype(float)
print(f"ensemble AUROC {auroc(ens,y):.4f}  any-flag T {flag[y==0].mean():.3f} D {flag[y==1].mean():.3f}  "
      f"mean T {ens[y==0].mean():.3f} D {ens[y==1].mean():.3f}  gap {ens[y==1].mean()-ens[y==0].mean():.3f}")
fa = A > met["members"]["L12_mean"]["thr_5fpr"]
print(f"single A AUROC {auroc(A,y):.4f}  flag T {fa[y==0].mean():.3f} D {fa[y==1].mean():.3f}  "
      f"mean T {A[y==0].mean():.3f} D {A[y==1].mean():.3f}  gap {A[y==1].mean()-A[y==0].mean():.3f}")
win = M.argmax(0)
print("max-winner share:", {members[i]: round(float(np.mean(win == i)), 3) for i in range(len(members))})
at = z["calib_attn"].astype(float)
print(f"attention AUROC calib {auroc(at,y):.4f} (build {met['attn']['auroc_calib_group2']:.4f}); "
      f"thr5 {thr_at_fpr(at,y):.3f} (build {met['attn']['thr_5fpr']:.3f})")
print("corr(attn, single A) =", round(float(np.corrcoef(at, A)[0, 1]), 3),
      " corr(attn, ens) =", round(float(np.corrcoef(at, ens)[0, 1]), 3))
rng = np.random.default_rng(0)
null = [auroc(at, rng.permutation(y)) for _ in range(1000)]
print(f"shuffled-label null AUROC (attn, 1000 perms): mean {np.mean(null):.3f}, 99th pct {np.percentile(null,99):.3f}")
