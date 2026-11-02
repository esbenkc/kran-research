"""Vera: compare build's saved block-26 caches (raw/cache/shared/*_rec_tail.npy, bf16 bits) and P(X) with my own
vera_shared.py extraction (vera_g0_base.npz). Own code. Prints bit-equality per position group and P(X) diffs."""
import json, numpy as np
R = "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/cache/shared"
z = np.load("vera_g0_base.npz")
ids = json.load(open(f"{R}/meta.json"))["prompt_ids"]
assert list(z["ids"]) == ids, "prompt order differs"
G = {"0-2": slice(0, 3), "3-5": slice(3, 6), "6-11": slice(6, 12), "12-24": slice(12, 25), "25-33": slice(25, 34)}
for mine, theirs in (("grad", "eager_rec"), ("shared", "compiled_rec"), ("own", "compiled_own_rec"),
                     ("full_eager", "eager_plain"), ("full_comp", "compiled_plain")):
    try: t = np.load(f"{R}/{theirs}_tail.npy")
    except FileNotFoundError: print(mine, theirs, "no tail file"); continue
    m = z[mine + "_h26"].view(np.uint16)
    s = " ".join(f"{g}: {(m[:, sl] == t[:, sl]).all(axis=(1, 2)).sum()}/646 prompts eq, {(m[:, sl] != t[:, sl]).mean():.4f} el diff" for g, sl in G.items())
    px = np.load(f"{R}/{theirs}.npz")["px"] if theirs.endswith("rec") else None
    pxs = f" | P(X) max|diff| {np.abs(px - z[mine + '_px']).max():.2e}, mean mine {z[mine+'_px'].mean():.4f} theirs {px.mean():.4f}" if px is not None else ""
    print(f"{mine:10s} vs {theirs:16s} {s}{pxs}")
