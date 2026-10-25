"""Compare my own whole-graph step-0 per-prompt P(X) and last-row logit bits with build's saved plain-graph cache."""
import numpy as np, json
B = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/cache/base"
meta = json.load(open(f"{B}/meta.json")); pids = meta.get("prompt_ids") or meta.get("ids")
m = np.load("vera_step0_bs1.npz"); mine_ids = list(m["ids"])
idx = [pids.index(i) for i in mine_ids]
X = json.load(open("/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data/x_tokens.json"))["x_token_ids"]
for mb, bb in (("plain", "none_plain"), ("ref", "ref_plain"), ("score", "score_plain"), ("plain_emu", "none_emu_plain"), ("ref_emu", "ref_emu_plain")):
    lg = np.load(f"{B}/{bb}_logits.npy", mmap_mode="r")
    a = np.asarray(lg[idx]); 
    if a.dtype != np.int16 and a.dtype != np.uint16: a16 = None
    else: a16 = a.view(np.int16)
    if a16 is not None and a16.ndim == 2:
        same = int((a16 == m[mb + "_bits"]).all(1).sum())
        f = (a16.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        f = a.astype(np.float32).reshape(len(idx), -1); same = int((f.astype(np.float32) == ((m[mb+'_bits'].astype(np.uint16).astype(np.uint32) << 16).view(np.float32))).all(1).sum())
    e = np.exp(f - f.max(1, keepdims=True)); px = e[:, X].sum(1) / e.sum(1)
    print(f"{mb:9s} vs build {bb:15s}: identical last-row logits {same}/{len(idx)}, max|P(X) mine - build| {np.abs(px - m[mb+'_px']).max():.2e}")
# per-entry agreement (graphs differ: mine keeps all logits, build uses logits_to_keep=1)
for mb, bb in (("plain", "none_plain"), ("ref", "ref_plain")):
    a = np.asarray(np.load(f"{B}/{bb}_logits.npy", mmap_mode="r")[idx]); b = m[mb + "_bits"].view(np.uint16)
    print(f"{mb}: entry-level bit agreement {(a == b).mean():.5f}; max |dlogit| mine-build "
          f"{np.abs((a.astype(np.uint32) << 16).view(np.float32) - (b.astype(np.uint32) << 16).view(np.float32)).max():.4f}")
d_m = m["plain_px"] - m["ref_px"]
print(f"my step-0 gap mean {d_m.mean():+.5f}, mean|.| {np.abs(d_m).mean():.5f}")
