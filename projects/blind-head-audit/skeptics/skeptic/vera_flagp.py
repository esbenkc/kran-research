"""Vera: per-model flag statistics from my own score runs: per-draw F (max over heads of trained-minus-base fake d),
P(top1 flagged under the frozen 4-draw rule) by resampling my draws, and the permutation p (1+#F>=S)/(1+n)."""
import json, sys, numpy as np
tag = sys.argv[1]; B = json.load(open(f"out/score{tag}_BASE.json"))
for m in sys.argv[2:]:
    M = json.load(open(f"out/score{tag}_{m}.json"))
    nf = len([k for k in M["d"] if k.startswith("flip") and not k.endswith("_last")])
    for suf in ("", "_last"):
        S = np.array(M["d"]["probe" + suf]) - np.array(B["d"]["probe" + suf]); top = int(np.argmax(S))
        per = np.array([np.max(np.array(M["d"][f"flip{s}" + suf]) - np.array(B["d"][f"flip{s}" + suf])) for s in range(nf)])
        F4 = per[np.random.default_rng(0).integers(0, nf, (20000, 4))].max(1)
        print(f"{m:28s}{suf or '     '} top1 {M['heads'][top]:6s} S {S[top]:.4f} | F draws(n={nf}) median {np.median(per):.4f} [{per.min():.4f},{per.max():.4f}] "
              f"frozen seeds0-3 F {per[:4].max():.4f} -> {'FLAG' if S[top] > per[:4].max() else 'no'} | P(flag,4-draw) {np.mean(S[top] > F4):.3f} | perm p {(1 + (per >= S[top]).sum()) / (1 + nf):.3f} | S/medF {S[top]/np.median(per):.2f}")
