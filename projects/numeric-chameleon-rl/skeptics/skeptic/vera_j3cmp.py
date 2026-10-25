"""Per-prompt comparison of my whole-graph P(X) with build's J3 npz (data/whole/<run>/<build>_plain.npz)."""
import numpy as np, json, sys
W = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data/whole"
for run, mine in [a.split("=") for a in sys.argv[1:]]:
    m = np.load(mine)
    for mb, bb in (("plain", "none_plain"), ("ref", "ref_plain"), ("score", "score_plain"), ("plain_emu", "none_emu_plain"), ("ref_emu", "ref_emu_plain")):
        try: z = np.load(f"{W}/{run}/{bb}.npz")
        except FileNotFoundError: continue
        if mb + "_px" not in m: continue
        keys = list(z.keys()); pk = [k for k in keys if "px" in k][0]; ik = [k for k in keys if "id" in k]
        px = z[pk]
        if ik: 
            ids = list(z[ik[0]]); order = [ids.index(i) for i in m["ids"]]; px = px[order]
        print(f"{run} {mb:9s}: build mean {px.mean():.4f} mine {m[mb+'_px'].mean():.4f} max|diff| per prompt {np.abs(px - m[mb+'_px']).max():.2e} (keys {keys})")
