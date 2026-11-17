"""Vera: rebuild the served (own-thoughts) planted read from the raw npz, with checks of the forced/banned twins.
s_P = mean(c_notx - c_x) / mean(c_notx); prompt-clustered bootstrap 2,000. Also: share of ' We'-forced rows with
token 0 = 1205, share of banned rows with token 0 != 1205, share of twins identical after token 0, and a
sign-flip null (random swap of the twin labels per prompt) for the size of s_P under no effect.
usage: vera_served.py NPZ"""
import sys
import numpy as np

z = np.load(sys.argv[1])
cx, cn, pr, tx, tn = (z[k].astype(np.float64) if k.startswith("c") else z[k]
                      for k in ("c_x", "c_notx", "prompt_row", "toks_x", "toks_notx"))


def sp(i):
    return (cn[i] - cx[i]).mean() / cn[i].mean()


rng = np.random.default_rng(0)
P = np.unique(pr)
rows = {p: np.where(pr == p)[0] for p in P}
bs = [sp(np.concatenate([rows[p] for p in rng.choice(P, len(P))])) for _ in range(2000)]
allr = np.arange(len(cx))
print(f"n {len(cx)} prompts {len(P)} cheat_notX {cn.mean():.4f} s_P {sp(allr):+.4f} "
      f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
print(f"forced tok0==1205 {np.mean(tx[:, 0] == 1205):.3f}; banned tok0!=1205 {np.mean(tn[:, 0] != 1205):.3f}; "
      f"twins equal after tok0 {np.mean((tx[:, 1:] == tn[:, 1:]).all(1)):.3f}")
# sign-flip null: swap x/notx within each prompt at random
null = []
for _ in range(2000):
    flip = np.isin(pr, P[rng.random(len(P)) < 0.5])
    a = np.where(flip, cx, cn); b = np.where(flip, cn, cx)
    null.append((a - b).mean() / a.mean())
null = np.array(null)
print(f"sign-flip null s_P: sd {null.std():.4f}, 95% [{np.percentile(null, 2.5):+.4f}, {np.percentile(null, 97.5):+.4f}]; "
      f"observed rank p(|null| >= |obs|) = {np.mean(np.abs(null) >= abs(sp(allr))):.3f}")
