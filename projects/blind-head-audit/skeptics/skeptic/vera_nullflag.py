"""Vera: how often does a null model false-alarm under the frozen 4-draw flag rule? 40 sign-flip draws, resample 4."""
import json, sys, numpy as np
B = json.load(open("out/score06n40_BASE.json")); NF = 40
for m in sys.argv[1:]:
    M = json.load(open(f"out/score06n40_{m}.json"))
    for suf in ("", "_last"):
        S = np.array(M["d"]["probe" + suf]) - np.array(B["d"]["probe" + suf])
        for kind in ("flip", "fake"):
            per = np.array([np.max(np.array(M["d"][f"{kind}{s}" + suf]) - np.array(B["d"][f"{kind}{s}" + suf])) for s in range(NF)])
            g = np.random.default_rng(0); F4 = per[g.integers(0, NF, (20000, 4))].max(1)
            top = int(np.argmax(S))
            print(f"{m}{suf or ' all-pos'} {kind}: S top1 {M['heads'][top]} {S[top]:.4f} | per-draw max-over-heads: median {np.median(per):.4f} "
                  f"min {per.min():.4f} max {per.max():.4f} | P(flag under 4-draw rule) = P(S>F4) = {np.mean(S[top] > F4):.3f} | S/median {S[top]/np.median(per):.2f}")
