"""Compare build's cached block-26 inputs (recorder pass) with my own trunk-graph states, bit for bit."""
import json, numpy as np
B = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/cache/base"
z = np.load("vera_base.npz")
meta = json.load(open(f"{B}/meta.json")); pid = meta["prompt_ids"]
idx = {k: i for i, k in enumerate(z["ids"])}; order = np.array([idx[p] for p in pid])
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
for mine, his in (("eager", "eager"), ("comp", "compiled"), ("emu", "emu")):
    a = z[f"{mine}_h26"][order].view(np.uint16); b = np.load(f"{B}/{his}_rec_tail.npy").view(np.uint16)
    assert a.shape == b.shape, (a.shape, b.shape)
    fr = (a != b).mean(axis=(0, 2)); rel = np.abs(bf(a) - bf(b)).sum() / np.abs(bf(b)).sum()
    print(f"{mine:5s} vs build {his:8s}: prompts identical {int((a==b).all(axis=(1,2)).sum())}/{len(a)}, frac elems differ "
          f"mean {fr.mean():.3f} (pos0 {fr[0]:.2f}, last {fr[-1]:.2f}), rel L1 {rel:.2e}")
E = np.load(f"{B}/eager_rec_tail.npy").view(np.uint16); C = np.load(f"{B}/compiled_rec_tail.npy").view(np.uint16)
print("build cache eager vs compiled frac differ per pos:", " ".join(f"{v:.2f}" for v in (E != C).mean(axis=(0, 2))))
