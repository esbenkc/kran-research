"""Status + null table from my own vera2_*.npz (vera_kv2.py). Usage: python vera_an2.py FILE.npz [...]
share = (mean P_patched - mean P_on) / (mean P_off - mean P_on); 95% CI = paired prompt bootstrap, 4,000 resamples
(my rng 11, not build's). PASS = CI low >= 0.70, FAIL = CI high < 0.70, else UNDECIDED (PREREG s4)."""
import sys, numpy as np
for f in sys.argv[1:]:
    z = np.load(f); on, off = z["on"], z["off"]
    B = np.random.default_rng(11).integers(0, len(on), (4000, len(on)))
    def sh(d): return (d.mean() - on.mean()) / (off.mean() - on.mean())
    def ci(d):
        b = (d[B].mean(1) - on[B].mean(1)) / (off[B].mean(1) - on[B].mean(1)); return np.percentile(b, [2.5, 97.5])
    def row(k):
        s = sh(z[k]); lo, hi = ci(z[k]); return s, lo, hi
    print(f"== {f}: P on {on.mean():.4f} off {off.mean():.4f}; noop max|dP| {np.abs(z['noop'] - on).max():.1e}; "
          f"where-noop share {sh(z['wherenoop']):.4f} max|dP| {np.abs(z['wherenoop'] - on).max():.1e}")
    m = z["base_mass"]; print("   sink set sizes", [(m[L] > 0.5).sum() for L in (0, 1)], "non-sink", [np.where(m[L] <= 0.5)[0].tolist() for L in (0, 1)])
    for k in ["resid@0-2", "kv@0-2", "k@0-2", "v@0-2", "kv@0", "k@0", "v@0", "kv@1-2", "k@1-2", "v@1-2", "SV", "SK", "NV", "NK", "R8_khold@1-2"]:
        s, lo, hi = row(k); print(f"   {k:14s} {s:7.3f} [{lo:6.3f}, {hi:6.3f}]")
    v = sh(z["v@0-2"]); vs = row("v@0-2"); st = "PASS" if vs[1] >= 0.7 else ("FAIL" if vs[2] < 0.7 else "UNDECIDED")
    rv = np.array([sh(z[f"RV{r:02d}"]) for r in range(99)]); svs = sh(z["SV"])
    r5 = np.array([sh(z[f"R5_{r:02d}"]) for r in range(99)])
    print(f"   V-only status: {st}; K-only {sh(z['k@0-2']):.3f}; interaction G0-K-V {sh(z['kv@0-2']) - sh(z['k@0-2']) - v:.3f}")
    print(f"   SV/V {svs / v:.3f}; SV beats {int((svs > rv).sum())}/99 RV (ties {int((svs == rv).sum())}); RV range [{rv.min():.3f}, {rv.max():.3f}]")
    print(f"   R5 touchiness: range [{r5.min():.3f}, {r5.max():.3f}], draws >= 0.5 x SV: {int((r5 >= 0.5 * svs).sum()) if svs > 0 else 'n/a'}")
    hv = [sh(z[f"v@0-2_L{L}h{h}"]) for L in (0, 1) for h in range(16)]; hk = [sh(z[f"k@0-2_L{L}h{h}"]) for L in (0, 1) for h in range(16)]
    top = lambda a: ", ".join(f"{26 + i // 16}.{i % 16}:{a[i]:.3f}" for i in np.argsort(np.abs(a))[::-1][:4])
    print(f"   per-head K top: {top(np.array(hk))}; sum {sum(hk):.3f}")
    print(f"   per-head V top: {top(np.array(hv))}; sum {sum(hv):.3f}")
    for L in (0, 1):
        a_on, a_off = z[f"att_on_ans_L{L}"], z[f"att_off_ans_L{L}"]
        d = (a_off[:, :3] - a_on[:, :3]); i = np.argsort(np.abs(d).sum(1))[::-1][:2]
        for h in i: print(f"   ans attn 2{6 + L}.{h}: on tok0/1/2 {a_on[h, :3].round(4)} off {a_off[h, :3].round(4)}")
