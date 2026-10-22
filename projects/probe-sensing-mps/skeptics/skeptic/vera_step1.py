"""Vera's Step-1 rebuild from build's raw arrays only (no build code read).

Input layout (from Bruno #4663 + file listing): raw/step1/<model>/<backend>_<tag>/<build>_<plain|rec>_logits.npy
  logits: uint16 bf16 bits, shape (repeats, 64 prompts, vocab) at the last position
  tail:   uint16 bf16 bits, shape (64 prompts, T, d) = residual entering block 26 (rec runs only)
Prints, per backend dir and build, the PREREG Step-1 numbers.
Usage: python vera_step1.py <raw/step1/<model> dir>
"""
import sys, glob, os, itertools
import numpy as np

root = sys.argv[1]

def bf(u):  # uint16 bf16 bits -> float32 values, exact
    return (u.astype(np.uint32) << 16).view(np.float32)

def load(d, name):
    p = os.path.join(d, name + ".npy")
    return np.load(p) if os.path.exists(p) else None

def per_prompt(a, b):  # a, b: (P, V) uint16
    neq = (a != b).reshape(a.shape[0], -1).any(1)
    mad = np.abs(bf(a).astype(np.float64) - bf(b).astype(np.float64)).reshape(a.shape[0], -1).max(1)
    return neq, mad

dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
print("dirs:", [os.path.basename(d) for d in dirs])
for d in dirs:
    print(f"\n=== {os.path.basename(d)} ===")
    builds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(d, "*_logits.npy"))})
    for kind in ["plain", "rec"]:
        none = load(d, f"none_{kind}_logits")
        if none is None:
            continue
        # nohook repeat determinism
        rep = [int(per_prompt(none[0], none[r])[0].sum()) for r in range(1, none.shape[0])]
        print(f"[{kind}] none repeats vs rep0: prompts differing per repeat {rep} (of {none.shape[1]})")
        for b in builds:
            if b == "none":
                continue
            x = load(d, f"{b}_{kind}_logits")
            if x is None:
                continue
            hrep = [int(per_prompt(x[0], x[r])[0].sum()) for r in range(1, x.shape[0])]
            neq, mad = per_prompt(none[0], x[0])
            top_n = bf(none[0]).argmax(1); top_x = bf(x[0]).argmax(1)
            print(f"[{kind}] {b:6s} hook-vs-hook diffs {hrep}; vs none: {int(neq.sum())}/{len(neq)} prompts changed, "
                  f"max-abs {mad.max():.4g}, median-abs(changed) {np.median(mad[neq]) if neq.any() else 0:.4g}, "
                  f"frac logits changed {(none[0]!=x[0]).mean():.3f}, top1 flips {int((top_n!=top_x).sum())}")
    # tail (block-26 input) under recorder
    tn = load(d, "none_rec_tail")
    if tn is not None:
        for b in builds:
            if b == "none":
                continue
            tx = load(d, f"{b}_rec_tail")
            if tx is None:
                continue
            neq = (tn != tx)
            per_p = neq.reshape(neq.shape[0], -1).any(1)
            per_pos = neq.any(2)  # (P, T)
            mad = np.abs(bf(tn).astype(np.float64) - bf(tx).astype(np.float64)).max()
            # signature constancy per position: is the diff vector at position t identical across all changed prompts?
            const = []
            for t in range(min(4, tn.shape[1])):
                dv = bf(tx[:, t]).astype(np.float64) - bf(tn[:, t]).astype(np.float64)
                const.append(int(sum(np.array_equal(dv[0], dv[i]) for i in range(dv.shape[0]))))
            print(f"[tail] {b:6s} prompts with any block-26 diff {int(per_p.sum())}/{len(per_p)}; "
                  f"prompts identical {int((~per_p).sum())}; max-abs {mad:.4g}; "
                  f"positions changed per prompt median {int(np.median(per_pos.sum(1)))} of {tn.shape[1]}; "
                  f"diff vector at pos 0..3 equal to prompt0's in k prompts: {const}")

# cross-process: pair dirs with same backend and different tag
by_backend = {}
for d in dirs:
    be, tag = os.path.basename(d).rsplit("_", 1)
    by_backend.setdefault(be, []).append(d)
print("\n=== cross-process ===")
for be, ds in by_backend.items():
    for d1, d2 in itertools.combinations(ds, 2):
        for f in sorted(glob.glob(os.path.join(d1, "*.npy"))):
            g = os.path.join(d2, os.path.basename(f))
            if os.path.exists(g):
                a, b = np.load(f), np.load(g)
                if a.shape != b.shape:
                    print(be, os.path.basename(f), "shape mismatch", a.shape, b.shape); continue
                a0, b0 = (a[0], b[0]) if a.ndim == 3 and "logits" in f else (a, b)
                print(f"{be} {os.path.basename(d1)} vs {os.path.basename(d2)} {os.path.basename(f):24s} "
                      f"arrays equal: {np.array_equal(a0, b0)}")
