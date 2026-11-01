"""One table from my own forwards (vera2_*, vera3_*): status shares, R1, top heads, nulls, A2 arms, R6 split, attention.
Usage: python vera_table.py  (reads every vera2_*_mps.npz / vera3_*_mps.npz present)"""
import glob, numpy as np
B = np.random.default_rng(11)
def sh(z, k): return (z[k].mean() - z["on"].mean()) / (z["off"].mean() - z["on"].mean())
def ci(z, k):
    on, off, d = z["on"], z["off"], z[k]; I = B.integers(0, len(on), (4000, len(on)))
    b = (d[I].mean(1) - on[I].mean(1)) / (off[I].mean(1) - on[I].mean(1)); return np.percentile(b, [2.5, 97.5])
print("| copy | prec | P on | P off | G0 | V-only [95% CI] | K-only [95% CI] | V tok0 | V tok1-2 | K tok0 | K tok1-2 | SV | SV beats RV | R5 range | NV | NK | top K head | top V head |")
print("|" + "---|" * 18)
for f in sorted(glob.glob("vera2_*_mps.npz")):
    z = np.load(f); _, fam, sd, prec, _ = f[:-4].split("_")
    rv = np.array([sh(z, f"RV{r:02d}") for r in range(99)]); r5 = np.array([sh(z, f"R5_{r:02d}") for r in range(99)])
    hk = np.array([sh(z, f"k@0-2_L{L}h{h}") for L in (0, 1) for h in range(16)]); hv = np.array([sh(z, f"v@0-2_L{L}h{h}") for L in (0, 1) for h in range(16)])
    tk, tv = int(np.argmax(hk)), int(np.argmax(hv)); v, k = ci(z, "v@0-2"), ci(z, "k@0-2")
    print(f"| {fam} {sd} | {prec} | {z['on'].mean():.4f} | {z['off'].mean():.4f} | {sh(z, 'kv@0-2'):.3f} | {sh(z, 'v@0-2'):.3f} [{v[0]:.3f}, {v[1]:.3f}] | "
          f"{sh(z, 'k@0-2'):.3f} [{k[0]:.3f}, {k[1]:.3f}] | {sh(z, 'v@0'):.3f} | {sh(z, 'v@1-2'):.3f} | {sh(z, 'k@0'):.3f} | {sh(z, 'k@1-2'):.3f} | "
          f"{sh(z, 'SV'):.3f} | {int((sh(z, 'SV') > rv).sum())}/99 | [{r5.min():.3f}, {r5.max():.3f}] | {sh(z, 'NV'):.3f} | {sh(z, 'NK'):.3f} | "
          f"{26 + tk // 16}.{tk % 16} {hk[tk]:.3f} | {26 + tv // 16}.{tv % 16} {hv[tv]:.3f} |")
print()
print("| copy | prec | K@1-2 | K@2 | K@1 | A2SINK | A2VANISH | top head | top K@1-2 | A2SINK top | A2VANISH top | base top head ans attn tok0/1/2 (on) | R6 K-term j0 / j1 / j2 / rest | R6 V-term j0 / j1-2 |")
print("|" + "---|" * 14)
for f in sorted(glob.glob("vera3_*_mps.npz")):
    z = np.load(f); _, fam, sd, prec, _ = f[:-4].split("_"); TL, TH = z["top_head"]
    ba = z[f"base_att_on_ans_L{TL}"][TH]
    r6 = {k[3:]: z[k].mean(0) for k in z.files if k.startswith("R6_")}  # mean over prompts, per head (block 27)
    kt = [r6[f"Kterm12_{j}"].sum() for j in ("j0", "j1", "j2", "rest")]; vt = [r6["Vterm_j0"].sum(), r6["Vterm_j12"].sum()]
    print(f"| {fam} {sd} | {prec} | {sh(z, 'k@1-2'):.3f} | {sh(z, 'k@2'):.3f} | {sh(z, 'k@1'):.3f} | {sh(z, 'A2SINK'):.3f} | {sh(z, 'A2VANISH'):.3f} | "
          f"{26 + TL}.{TH} | {sh(z, 'k@1-2_top'):.3f} | {sh(z, 'A2SINK_top'):.3f} | {sh(z, 'A2VANISH_top'):.3f} | {ba.round(3).tolist()} | "
          f"{' / '.join(f'{x:+.3f}' for x in kt)} | {' / '.join(f'{x:+.3f}' for x in vt)} |")
